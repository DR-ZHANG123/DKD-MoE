"""01_size_estimate: count and size the KPMP open files each dkd_moe category needs, per participant group.

Writes dkd_moe/results/data_inventory/size_estimate.tsv (category x group: files, participants, GB in the
repository, GB still to fetch) and download_plan_<category>.tsv (files to fetch; local copies from the
main pipeline are reused). The ATAC-Seq "Expression Matrix" archives are recorded but not planned: each
holds the same fragments + filtered h5 as the 10x Multiome archive of the same library (verified on a probe pair)."""
from __future__ import annotations
import pandas as pd
from dcommon import cfg, select, file_table, groups, existing_raw_names, section_of, rpath, result_dir, write_provenance, CATEGORIES


def visium_needed(V: pd.DataFrame, c: dict, local: dict) -> pd.Series:
    done_sections = {p.name for p in rpath(c["existing_visium"]).glob("*/*") if (p / "outs/filtered_feature_bc_matrix.h5").exists()}
    need = []
    for r in V.itertuples():
        if r.file_name in local:
            need.append(False)
        elif r.file_name.endswith(".tar.gz"):
            need.append(section_of(r.file_name) not in done_sections)
        else:
            need.append(True)
    return pd.Series(need, index=V.index)


def main():
    c = cfg()
    F = file_table(c)
    G = groups().group
    local = existing_raw_names(c)
    out = result_dir("data_inventory")
    rows, plans = [], {}
    for cat in CATEGORIES + ["atac_seq"]:
        S = select(cat, c, F)
        if cat == "visium":
            S["needed"] = visium_needed(S, c, local)
        elif cat == "atac_seq":
            S["needed"] = False
        else:
            S["needed"] = ~S.file_name.isin(local)
        S["group"] = "atlas (multi-participant)" if cat == "annotation" else S.redcap_id.map(G).fillna("not_in_clinical_table")
        plans[cat] = S
        for grp, d in list(S.groupby("group")) + [("ALL", S)]:
            rows.append({"category": cat, "group": grp, "n_files": len(d), "n_participants": d.redcap_id.nunique(),
                         "repo_GB": round(d.file_size.sum() / 1e9, 2), "n_to_fetch": int(d.needed.sum()),
                         "fetch_GB": round(d.loc[d.needed, "file_size"].sum() / 1e9, 2)})
    T = pd.DataFrame(rows)
    cap = c["size_cap_tb"] * 1e3
    for cat, S in plans.items():
        tot = T[(T.category == cat) & (T.group == "ALL")].fetch_GB.iloc[0]
        restrict = tot > cap
        T.loc[T.category == cat, "restricted_to_core_groups"] = restrict
        P = S[S.needed & (S.group.isin(c["core_groups"]) if restrict else S.needed)]
        if cat != "atac_seq":
            P.to_csv(out / f"download_plan_{cat}.tsv", sep="\t", index=False)
    T.to_csv(out / "size_estimate.tsv", sep="\t", index=False)
    print(T[T.group == "ALL"].to_string(index=False))
    write_provenance("data_inventory", [rpath(c["file_table"]), rpath(c["clinical_table"])],
                     [out / "size_estimate.tsv"] + [out / f"download_plan_{k}.tsv" for k in CATEGORIES], c["seed"], cfg=c,
                     extra={"note": "ATAC-Seq archives duplicate the 10x Multiome archives (fragments + filtered h5); not fetched"})


if __name__ == "__main__":
    main()
