"""Assemble Tables 1-10 using the reference paper's experiment taxonomy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "tmp" / "reference_interval_only_5000_seed20260507_20260822"
MERGED = RUN / "merged_results"
MISSING = RUN / "reference_exact_missing"
OUT = RUN / "reference_exact_tables"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(name: str) -> pd.DataFrame:
    return pd.read_csv(MERGED / f"{name}_5000_summary.csv")


def missing_summary(group: str) -> pd.DataFrame:
    paths = sorted((MISSING / group / "revision_results").glob(f"{group}_interval5000*_summary.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one complete 5000 summary for {group}, got {paths}")
    return pd.read_csv(paths[0])


def select(df: pd.DataFrame, model: str, agents: tuple[int, ...] = (10, 40, 100)) -> pd.DataFrame:
    result = df[(df["Model"] == model) & df["Agents"].isin(agents)].copy()
    if set(result["Agents"]) != set(agents):
        raise RuntimeError(f"Incomplete model rows for {model}: {result[['Model', 'Agents']].to_dict('records')}")
    return result.sort_values("Agents")


def common_rows(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "模型": label,
            "决策者数量": frame["Agents"].astype(int),
            "共识达成率": frame["Success"],
            "平均收敛步数": frame["Steps"],
            "总调整成本": frame["Total Cost"],
            "成本基尼系数": frame["Cost Gini"],
            "边界违规次数": frame["Boundary Violations"],
            "平均回合奖励": frame["Total Reward"],
        }
    )


def write(number: int, slug: str, frame: pd.DataFrame, outputs: list[dict]) -> None:
    path = OUT / f"table{number:02d}_{slug}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    outputs.append({"number": number, "path": str(path), "rows": len(frame), "sha256": sha256(path)})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    outputs: list[dict] = []
    table1 = pd.DataFrame(
        [
            ["偏好", "中心域/半宽", "[0.1+h,0.9-h] / U[0.02,0.08]", "区间端点保持在安全域"],
            ["环境", "训练/测试成员数", "10 / 10,40,100", "仅N=10训练"],
            ["环境", "共识阈值/最大轮数", "0.90 / 120", "与参考设置一致"],
            ["成本", "系统成本系数/幂次", "75 / 2", "基于实际中心移动"],
            ["奖励", "共识/边界/离群/公平", "50/20/2/1", "改为区间端点口径"],
            ["网络", "状态/动作/token/z", "5/1/3/8", "标量动作平移整个区间"],
            ["网络", "历史窗口/Transformer层/头", "10/2/4", "参考设置"],
            ["网络", "Actor/Critic隐藏层", "256", "成员共享"],
            ["SAC", "γ/τ/α/KL权重", "0.95/0.01/0.1/0.05", "参考设置"],
            ["优化", "学习率/batch/回放池", "3e-4/32/100000", "Adam"],
            ["训练", "回合/warmup", "5000/150", "所有可训练变体"],
            ["评估", "每规模场景数", "50", "固定环境种子"],
        ],
        columns=["类别", "参数", "取值", "说明"],
    )
    write(1, "parameters", table1, outputs)

    rules = summary("rules")
    main = summary("main")
    rule_map = [
        ("Bounded confidence", "有限信任"),
        ("Mean consensus", "均值趋同"),
        ("Random", "随机"),
        ("Cost-aware heuristic", "成本感知"),
    ]
    table2 = pd.concat(
        [common_rows(select(rules, model), label) for model, label in rule_map]
        + [common_rows(main.sort_values("Agents"), "区间Meta-GDM")],
        ignore_index=True,
    )
    write(2, "rule_baselines", table2, outputs)

    rl = summary("rl")
    table3 = pd.concat(
        [common_rows(select(rl, model, (10, 40)), model) for model in ("PPO", "DDPG", "SAC")]
        + [common_rows(main[main["Agents"].isin((10, 40))].sort_values("Agents"), "区间Meta-GDM")],
        ignore_index=True,
    )
    write(3, "rl_baselines", table3, outputs)

    core = summary("core")
    core_map = [
        ("Full Meta-GDM", "完整模型"),
        ("w/o both", "去上下文与社会注意力"),
        ("w/o context encoder", "去上下文"),
        ("w/o social attention", "去社会注意力"),
    ]
    table4 = pd.concat(
        [common_rows(select(core, model), label) for model, label in core_map], ignore_index=True
    )
    write(4, "core_ablation", table4, outputs)

    attention = summary("attention")
    attention_map = [
        ("mean_pooling", "均值池化"),
        ("social_attention", "社会注意力"),
        ("standard_mha", "标准多头注意力"),
    ]
    table5 = pd.concat(
        [common_rows(select(attention, model), label) for model, label in attention_map], ignore_index=True
    )
    write(5, "attention_replacement", table5, outputs)

    table6 = pd.DataFrame(
        [
            ["PPO", "3e-4", 0.95, 256, 0.2, "—", "—", "全批", "—"],
            ["DDPG", "3e-4", 0.95, 256, "—", 0.01, "—", 64, 100000],
            ["SAC", "3e-4", 0.95, 256, "—", 0.01, 0.1, 64, 100000],
            ["区间Meta-GDM", "3e-4", 0.95, 256, "—", 0.01, 0.1, 32, 100000],
        ],
        columns=["基线模型", "学习率", "折扣因子", "隐藏层维度", "clip系数", "目标网络软更新系数", "熵系数", "Batch size", "经验回放池容量"],
    )
    write(6, "rl_hyperparameters", table6, outputs)

    latent = pd.concat([summary("latent"), missing_summary("latent_reference")], ignore_index=True)
    table7_parts = []
    for z in (4, 8, 16, 32, 64):
        model = f"z={z}"
        frame = select(latent, model)
        part = common_rows(frame, str(z)).rename(columns={"模型": "潜变量维度"})
        part["最终共识水平"] = frame["Final Consensus"].to_numpy()
        table7_parts.append(part)
    table7 = pd.concat(table7_parts, ignore_index=True)
    write(7, "latent_dimension", table7, outputs)

    reward = summary("reward")
    cost_extra = missing_summary("cost_reference")
    cost = pd.concat([reward[reward["Model"].str.startswith("cost=")], cost_extra], ignore_index=True)
    table8_parts = []
    for coefficient in (25, 50, 75, 100, 125):
        frame = select(cost, f"cost={coefficient}")
        part = pd.DataFrame(
            {
                "系统成本系数": coefficient,
                "决策者数量": frame["Agents"].astype(int),
                "共识达成率": frame["Success"],
                "平均收敛步数": frame["Steps"],
                "成本基尼系数": frame["Cost Gini"],
                "边界违规次数": frame["Boundary Violations"],
                "高度归一成本": frame["Total Cost"] / coefficient,
            }
        )
        table8_parts.append(part)
    table8 = pd.concat(table8_parts, ignore_index=True)
    write(8, "cost_coefficient", table8, outputs)

    weight_extra = missing_summary("consensus_weight_reference")
    table9_parts = []
    for weight in (30, 40, 50, 60, 70):
        frame = select(weight_extra, f"consensus={weight}")
        part = common_rows(frame, str(weight)).rename(columns={"模型": "共识增益权重"})
        table9_parts.append(part)
    table9 = pd.concat(table9_parts, ignore_index=True)
    write(9, "consensus_gain_weight", table9, outputs)

    decay_path = OUT / "decay_dynamics_source.csv"
    if not decay_path.is_file():
        raise FileNotFoundError(
            f"Run evaluate_reference_consensus_decay_table.py first; missing {decay_path}"
        )
    decay = pd.read_csv(decay_path)
    table10 = decay.rename(
        columns={
            "consensus_decay": "共识衰减系数",
            "agents": "决策者数量",
            "success": "共识达成率",
            "steps": "平均收敛步数",
            "cost_gini": "成本基尼系数",
            "boundary_violations": "边界违规次数",
            "initial_consensus": "初始共识水平",
            "final_consensus": "最终共识水平",
            "initial_preference_std": "初始偏好标准差",
            "final_preference_std": "最终偏好标准差",
        }
    )[
        [
            "共识衰减系数",
            "决策者数量",
            "共识达成率",
            "平均收敛步数",
            "成本基尼系数",
            "边界违规次数",
            "初始共识水平",
            "最终共识水平",
            "初始偏好标准差",
            "最终偏好标准差",
        ]
    ]
    write(10, "consensus_decay", table10, outputs)

    manifest = {
        "schema": "reference_exact_interval_tables_v1",
        "table_count": 10,
        "train_episodes_for_all_trainable_rows": 5000,
        "outputs": outputs,
        "source_summaries": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted(MERGED.glob("*_5000_summary.csv"))
        },
        "omitted_extra_experiments": [
            "cluster consensus and MCCM (not in the reference paper's displayed rule table)",
            "context-form, boundary-weight, history-window, head-count, and learning-rate diagnostics",
        ],
    }
    manifest_path = OUT / "table_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "tables": len(outputs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
