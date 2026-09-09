from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

from extraction.backends.qwen_penalty import PENALTY_KEY
from extraction.evidence import load_images
from extraction.qwen_config import qwen_settings


@dataclass
class QwenOutput:
    text: str
    prompt_tokens: int
    output_tokens: int


@dataclass
class QwenBackend:
    engine: Any
    processor: Any
    model_id: str
    executor: ThreadPoolExecutor
    stop_token_ids: list[int]
    runtime_info: dict[str, Any]

    @classmethod
    def from_pretrained(cls, model_path: str, *, settings=None, image_limit=6):
        try:
            import torch
            from transformers import AutoProcessor
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM
        except ImportError as exc:
            raise RuntimeError("Qwen requires Linux/CUDA and the 'qwen' dependencies (vllm==0.28.0)") from exc
        if version("vllm").split("+")[0] != "0.28.0":
            raise RuntimeError("Qwen backend requires vllm==0.28.0")
        model_dir = Path(model_path)
        if not model_dir.is_dir():
            raise ValueError(f"Qwen local checkpoint directory is missing: {model_path}")
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        if config.get("quantization_config"):
            raise ValueError("Qwen vLLM pipeline requires a non-quantized BF16 checkpoint")
        resolved = qwen_settings(settings)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        generation_path = model_dir / "generation_config.json"
        generation = json.loads(generation_path.read_text(encoding="utf-8")) if generation_path.exists() else {}
        eos = generation.get("eos_token_id", processor.tokenizer.eos_token_id)
        stop_ids = list(eos) if isinstance(eos, list) else ([eos] if eos is not None else [])
        arguments = {
            **resolved,
            "model": model_path,
            "model_impl": "vllm",
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
            "tensor_parallel_size": 1,
            "generation_config": "vllm",
            "limit_mm_per_prompt": {"image": image_limit, "video": 0},
            "logits_processors": ["extraction.backends.qwen_logits:GeneratedTokenLogitsProcessor"],
            "enable_log_requests": False,
            "disable_log_stats": True,
        }
        if arguments["max_model_len"] == "auto":
            arguments["max_model_len"] = -1
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**arguments))
        try:
            actual = {
                **resolved,
                "max_model_len": engine.model_config.max_model_len,
                "async_scheduling": engine.vllm_config.scheduler_config.async_scheduling,
                "dtype": "bfloat16",
                "kv_cache_dtype": "auto",
                "tensor_parallel_size": 1,
                "image_limit": image_limit,
                "stop_token_ids": stop_ids,
                "model_impl": "vllm",
                "generation_config": "vllm",
                "logits_processors": arguments["logits_processors"],
            }
            info = {
                "backend": "vllm",
                "versions": {name: version(name) for name in ("vllm", "torch", "transformers")},
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "settings": actual,
            }
            return cls(engine, processor, model_path,
                       ThreadPoolExecutor(max_workers=resolved["renderer_num_workers"]), stop_ids, info)
        except BaseException:
            engine.shutdown()
            raise

    def _prepare(self, task):
        images = load_images(list(task.image_paths))
        try:
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": task.prompt})
            prompt = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
            )
            inputs = {"prompt": prompt}
            if images:
                inputs["multi_modal_data"] = {"image": images}
            return inputs, images
        except BaseException:
            for image in images:
                image.close()
            raise

    def sampling_params(self, task):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        if task.do_sample and any(v is None for v in (task.temperature, task.top_p, task.top_k)):
            raise ValueError("sampled Qwen generation requires temperature, top_p, and top_k")
        return SamplingParams(
            n=1,
            temperature=task.temperature if task.do_sample else 0.0,
            top_p=task.top_p if task.do_sample else 1.0,
            top_k=task.top_k if task.do_sample else 0,
            max_tokens=task.max_new_tokens,
            seed=task.seed,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            extra_args={PENALTY_KEY: task.repetition_penalty},
            stop_token_ids=self.stop_token_ids,
            ignore_eos=False,
            skip_special_tokens=True,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

    async def generate(self, task) -> QwenOutput:
        from vllm.renderers.params import TokenizeParams

        images = []
        try:
            params = self.sampling_params(task)
            preparation = asyncio.get_running_loop().run_in_executor(self.executor, self._prepare, task)
            try:
                inputs, images = await asyncio.shield(preparation)
            except asyncio.CancelledError:
                # A running image-loader thread cannot be cancelled; release its eventual images.
                def release(future):
                    if not future.cancelled() and future.exception() is None:
                        for image in future.result()[1]:
                            image.close()
                preparation.add_done_callback(release)
                raise
            # Render once so expanded image tokens are counted before admission.
            # Never truncate the prompt or reduce max_tokens to fit.
            rendered = (await self.engine.renderer.render_cmpl_async(
                [inputs], tok_params=TokenizeParams(max_total_tokens=None, add_special_tokens=False),
            ))[0]
            prompt_length = len(rendered["prompt_token_ids"])
            required = prompt_length + task.max_new_tokens
            maximum = self.engine.model_config.max_model_len
            if required > maximum:
                raise ValueError(f"context length required={required} (input={prompt_length}, "
                                 f"output={task.max_new_tokens}) exceeds max_model_len={maximum}")
            final = None
            async for output in self.engine.generate(rendered, params, task.task_id):
                if output.finished:
                    final = output
            if final is None or len(final.outputs) != 1:
                raise RuntimeError("vLLM did not return one completed output")
            return QwenOutput(final.outputs[0].text.strip(), prompt_length, len(final.outputs[0].token_ids))
        except Exception as exc:
            raise RuntimeError(f"Qwen task {task.task_id}: {exc}") from exc
        finally:
            for image in images:
                image.close()

    def close(self):
        try:
            self.engine.shutdown(timeout=5)
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
