"""Korean report from recorded recommendation definitions and all paired comparisons."""


def display_arm(name):
    return name


def write_report(context, selected, document):
    report = document["recommendations"]
    means, deviations = report["means"], report["standard_deviations"]
    statistics = document["statistics"]
    relative = report["base_relative_changes"]
    baseline = report["baseline"]
    definitions = {row["label"]: row for row in document["experiments"]}
    lines = [
        f"# {context.run_id} / {context.dataset}: 추천 실험", "",
        f"계약: `{document['arm_contract']}`. 모델 구조: `{report['architecture_version']}`.", "",
        "| label | source | modules | readout | 모듈 버전 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in definitions.values():
        versions = ", ".join(f"{k}={v}" for k, v in row["module_versions"].items()) or "—"
        lines.append(f"| {row['label']} | {row['source']} | {', '.join(row['modules']) or '—'} "
                     f"| {row['readout']} | {versions} |")
    if context.dataset == "m3l_10m":
        from preparation.m3l import INTERPRETATION

        lines += ["", f"{INTERPRETATION}. 모든 평점을 사용하며 영상 미확보 영화는 제외합니다."]
    lines += ["", f"상대 변화율 기준: `{baseline}`." if baseline else "metadata 기준 실험이 없어 상대 변화율을 생략합니다.", "",
              "| label | NDCG@10 평균 ± 표준편차 | HR@10 평균 ± 표준편차 | 기준 대비 NDCG | 기준 대비 HR |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for arm, values in means.items():
        changes = [
            f"{relative[arm][metric] * 100:+.2f}%"
            if relative and relative[arm][metric] is not None else "—"
            for metric in ("NDCG@10", "HR@10")
        ]
        lines.append(
            f"| {arm} | {values['NDCG@10']:.6f} ± {deviations[arm]['NDCG@10']:.6f} "
            f"| {values['HR@10']:.6f} ± {deviations[arm]['HR@10']:.6f} "
            f"| {changes[0]} | {changes[1]} |"
        )
    lines += [
        "", "사용자별 seed 평균 NDCG@10에 paired bootstrap을 적용합니다. "
        f"실험의 모든 {statistics['comparison_count']}쌍에 Bonferroni 보정을 적용합니다. "
        "HR@10의 통계적 유의성 검정은 수행하지 않습니다.", "",
        "| 비교 (왼쪽 − 오른쪽) | NDCG 차이 | 보정 신뢰구간 | 해석 |",
        "| --- | ---: | --- | --- |",
    ]
    for value in statistics["comparisons"].values():
        left, right = value["left"], value["right"]
        delta, low, high = value["difference"], value["ci_low"], value["ci_high"]
        interpretation = "우위 근거 확보" if low > 0 else "열위 근거 확보" if high < 0 else "차이 불확실"
        lines.append(
            f"| {left} − {right} | {delta:+.6f} "
            f"| [{low:+.6f}, {high:+.6f}] ({value['confidence_level']:.4%}) "
            f"| {interpretation} |"
        )
    lines += ["", "| label | 소스 | 유효 장면 | 장면 성공률 | 제목 fallback | 제목·장면 모두 없음 | 학습 가능 파라미터 | 실제 사용 파라미터 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, state in document["representations"].items():
        counts = state["statistics"]["counts"]
        coverage = state["statistics"]["scene_coverage"]
        coverage_text = f"{coverage:.2%}" if coverage is not None else "—"
        parameters = report["model_reports"][arm]["parameters"]
        source = ", ".join(row["source"] for row in state["sources"])
        empty = counts.get("empty_video", 0)
        fallback = max(0, counts.get("zero_scene_video", 0) - empty) if definitions[arm]["source"] != "metadata" else "—"
        lines.append(
            f"| {arm} | {source} | {counts.get('valid_scenes', 0)} | {coverage_text} "
            f"| {fallback} | {empty} | {parameters.get('trainable', '—')} "
            f"| {parameters.get('active_trainable', '—')} |"
        )
    lines += [
        "", "비교 결과는 위에 기록한 입력·모듈 구성 전체의 차이입니다. "
        "여러 조건이 다르면 개별 모듈만의 효과로 해석할 수 없습니다.",
        "Description과 Graph는 각각의 유효 장면을 사용하므로 장면 확보율도 함께 해석해야 합니다.",
        "", "| label | seed | NDCG@10 | HR@10 |", "| --- | ---: | ---: | ---: |",
    ]
    for row in report["per_seed"]:
        lines.append(f"| {row['arm']} | {row['seed']} | {row['NDCG@10']:.6f} "
                     f"| {row['HR@10']:.6f} |")
    destination = context.run_root / "reports" / selected / "report.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(destination)
