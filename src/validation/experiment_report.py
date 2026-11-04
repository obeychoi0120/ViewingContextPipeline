"""Korean report for manually selected, matched recommendation experiments."""

PRIORITY_COMPARISONS = (
    ("g0", "d", "Description과 구조화된 Graph 입력·인코딩"),
    ("g1", "g0", "A: Hypergraph 집계·Action 구간 기여의 추가 효과 (Role Fusion 없음)"),
    ("g2", "g1", "B: Role Fusion의 추가 효과 (Hypergraph 적용 상태)"),
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
        f"계약: `{document['arm_contract']}`. 모델 구조: `{report['architecture_version']}`. "
        f"Arm: {', '.join(document['targets'])}.", "",
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
    lines += ["", "| Arm | 소스 | 유효 장면 | 장면 성공률 | 제목 fallback | 제목·장면 모두 없음 | 학습 가능 파라미터 | 실제 사용 파라미터 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, state in document["representations"].items():
        counts = state["statistics"]["counts"]
        coverage = state["statistics"]["scene_coverage"]
        coverage_text = f"{coverage:.2%}" if coverage is not None else "—"
        parameters = report["model_reports"][arm]["parameters"]
        source = ", ".join(row["source"] for row in state["sources"])
        empty = counts.get("empty_video", 0)
        fallback = max(0, counts.get("zero_scene_video", 0) - empty) if arm != "base" else "—"
        lines.append(
            f"| {display_arm(arm)} | {source} | {counts.get('valid_scenes', 0)} | {coverage_text} "
            f"| {fallback} | {empty} | {parameters.get('trainable', '—')} "
            f"| {parameters.get('active_trainable', '—')} |"
        )
    lines += [
        "", "G0−D에는 추출 내용·인코더·장면 확보율 차이가 포함됩니다. "
        "장면 간 전달은 G1·G2의 Hypergraph에서 수행하며 GCN은 사용하지 않습니다. "
        "G1−G0는 Role Fusion 없이 Hypergraph 전달과 구간 기여를 추가한 A 전체의 효과입니다. "
        "G2−G1은 Hypergraph가 있는 상태에서 Role Fusion(B)을 추가한 효과입니다. "
        "파라미터 증가도 포함하며 A의 개별 경로 또는 연결 구조만의 효과를 분리하지 않습니다. "
        "각 효과를 단순 합산하지 않습니다.",
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
