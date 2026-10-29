"""Korean report for manually selected, matched recommendation experiments."""

PRIORITY_COMPARISONS = (
    ("g1", "d1", "Graph와 Description 파이프라인"),
    ("g1", "g1a", "Entity–Role Fusion 모듈"),
    ("g1", "g1b", "추출된 역할 연결 유지"),
    ("g1", "g1c", "학습된 장면 간 정보 전달"),
    ("g1c", "g0", "장면별 추가 변환층"),
    ("g1", "g0", "전체 장면 집계 모듈"),
)


def display_arm(name):
    return "Base" if name == "base" else name[0].upper() + name[1:]


def write_report(context, selected, document):
    report = document["recommendations"]
    means, deviations = report["means"], report["standard_deviations"]
    statistics = document["statistics"]
    relative = report["base_relative_changes"]
    lines = [
        f"# {context.run_id} / {context.dataset}: 추천 실험", "",
        f"계약: `{document['arm_contract']}`. Arm: {', '.join(document['targets'])}.", "",
        "| Arm | NDCG@10 평균 ± 표준편차 | HR@10 평균 ± 표준편차 | Base 대비 NDCG | Base 대비 HR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if context.dataset == "m3l_10m":
        from preparation.m3l import INTERPRETATION

        lines[2:2] = [f"{INTERPRETATION}. 모든 평점을 사용하며 영상 미확보 영화는 제외합니다.", ""]
    for arm, values in means.items():
        changes = [
            f"{relative[arm][metric] * 100:+.2f}%"
            if relative and relative[arm][metric] is not None else "—"
            for metric in ("NDCG@10", "HR@10")
        ]
        label = display_arm(arm)
        lines.append(
            f"| {label} | {values['NDCG@10']:.6f} ± {deviations[arm]['NDCG@10']:.6f} "
            f"| {values['HR@10']:.6f} ± {deviations[arm]['HR@10']:.6f} "
            f"| {changes[0]} | {changes[1]} |"
        )
    lines += [
        "", "사용자별 seed 평균 NDCG@10에 paired bootstrap을 적용합니다. "
        f"선택한 Arm의 모든 {statistics['comparison_count']}쌍에 Bonferroni 보정을 적용합니다. "
        "HR@10의 통계적 유의성 검정은 수행하지 않습니다.", "",
        "| 비교 | 검증 대상 | NDCG 차이 | 보정 신뢰구간 | 해석 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    tests = statistics["comparisons"]
    for left, right, purpose in PRIORITY_COMPARISONS:
        value = tests.get(f"{left}-{right}")
        reverse = False
        if value is None:
            value, reverse = tests.get(f"{right}-{left}"), True
        if value is None:
            continue
        delta, low, high = value["difference"], value["ci_low"], value["ci_high"]
        if reverse:
            delta, low, high = -delta, -high, -low
        interpretation = "우위 근거 확보" if low > 0 else "열위 근거 확보" if high < 0 else "차이 불확실"
        lines.append(
            f"| {display_arm(left)} − {display_arm(right)} | {purpose} | {delta:+.6f} "
            f"| [{low:+.6f}, {high:+.6f}] ({value['confidence_level']:.4%}) "
            f"| {interpretation} |"
        )
    lines += ["", "| Arm | 유효 장면 | 장면 성공률 | 장면 미확보 영상 | 학습 가능 파라미터 | 실제 사용 파라미터 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    corruption_lines = []
    for arm, state in document["representations"].items():
        counts = state["statistics"]["counts"]
        coverage = state["statistics"]["scene_coverage"]
        coverage_text = f"{coverage:.2%}" if coverage is not None else "—"
        parameters = report["model_reports"][arm]["parameters"]
        lines.append(
            f"| {display_arm(arm)} | {counts.get('valid_scenes', 0)} | {coverage_text} "
            f"| {counts.get('zero_scene_video', 0)} | {parameters.get('trainable', '—')} "
            f"| {parameters.get('active_trainable', '—')} |"
        )
        corruption = report["model_reports"][arm]["role_corruption"]
        if corruption:
            corruption_lines += ["", f"{display_arm(arm)}: 연결 {corruption['total_links']}개 중 "
                                 f"{corruption['changed_links']}개 변경 ({corruption['changed_ratio']:.2%}), "
                                 f"대체 Entity가 없는 연결 {corruption['impossible_links']}개. 교란 seed는 42입니다."]
    lines += corruption_lines
    lines += [
        "", "G1a 비교에는 Fusion 제거에 따른 모델 용량 변화가 포함됩니다. "
        "G1b는 추출된 연결을 유지하는 효과이며 원본 연결의 정답 여부를 보장하지 않습니다. "
        "G1c의 Q/K는 forward와 gradient에 관여하지 않습니다. 각 효과를 단순 합산하지 않습니다.",
        "Description과 Graph는 각각의 유효 장면을 사용하므로 장면 확보율도 함께 해석해야 합니다.",
        "", "| Arm | seed | NDCG@10 | HR@10 |", "| --- | ---: | ---: | ---: |",
    ]
    for row in report["per_seed"]:
        lines.append(f"| {display_arm(row['arm'])} | {row['seed']} | {row['NDCG@10']:.6f} "
                     f"| {row['HR@10']:.6f} |")
    destination = context.run_root / "reports" / selected / "report.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(destination)
