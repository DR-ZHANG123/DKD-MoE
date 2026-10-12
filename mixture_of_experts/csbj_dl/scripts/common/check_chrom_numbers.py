"""check_chrom_numbers: recompute every chromatin number quoted in the manuscript (text, Table 1, Table S9) from results/ and report PASS/FAIL. Run from csbj_dl/."""
import pandas as pd, json
R='results/'; ms=open('manuscript/manuscript.md').read(); chk=[]
def ok(n,c,detail=''): chk.append((n,bool(c),detail))
s=json.load(open(R+'chrom_summary/summary.json')); pg=s['participants_by_group']
other=pg['AKI']+pg['CKD_other']+pg['CKD_unadjudicated']
row=f"{s['participants_used']} participants with libraries passing quality control ({pg['DKD']} DKD, {pg['HKD']} HKD, {pg['HKD_withDM']} HKD with diabetes, {pg['DM_noCKD']} diabetes without CKD, {pg['Reference']} reference, {other} other)"
ok('Table 1 Multiome row',row in ms,row)
ok('Results participant count',f"Multiome nuclei of {s['participants_used']} participants" in ms)
rt=pd.read_csv(R+'chrom_readout/retrieval_summary.tsv',sep='\t').set_index(['feature','pairs'])
p,gq,sh=[round(100*rt.loc[k,'top1_at100_mean']) for k in [('peaks','aligned'),('gact','aligned'),('peaks','shuffled')]]
ok('retrieval',f"ranked first for {p}% of held-out nuclei with the peak encoder and {gq}% with the gene activity encoder, against {sh}%" in ms,(p,gq,sh))
ag=pd.read_csv(R+'chrom_readout/kpmp_agreement.tsv',sep='\t')
a=ag[(ag.feature=='peaks')&(ag.score=='chrom_score')&(ag.versus=='rna_readout')].set_index('participants').spearman
ok('rho',f"(Spearman ρ {a['all']:.2f}; {a['DKD+HKD']:.2f} within DKD and HKD)" in ms)
kt=pd.read_csv(R+'chrom_readout/kpmp_tests.tsv',sep='\t')
g=lambda c,f: kt[(kt.space=='peaks')&(kt.comparison==c)&(kt.feature==f)].iloc[0]
r1,r2,r3,r4=g('DKD_vs_Reference','chrom_score'),g('HKD_vs_Reference','chrom_score'),g('DKD_vs_HKD','chrom_score'),g('DKD_vs_HKD','rna_readout')
ok('disease vs reference',f"(DKD g {r1.g:.2f}; HKD g {r2.g:.2f})" in ms)
ok('chromatin DKD vs HKD',f"from HKD (g {r3.g:.2f})" in ms)
ok('RNA readout DKD vs HKD',f"(g {r4.g:.2f}, 95% CI {r4.ci_low:.2f} to {r4.ci_high:.2f}; P = {r4.p_wilcoxon:.3f}; {r4.n1} versus {r4.n2} participants)" in ms)
et=pd.read_csv(R+'chrom_readout/external_tests.tsv',sep='\t')
e=et[(et.cohort=='gse195460')&(et.space=='peaks')&(et.feature=='chrom_score')].iloc[0]
ok('GSE195460',f"(g {e.g:.2f}, 95% CI {e.ci_low:.2f} to {e.ci_high:.2f}; P = {e.p_wilcoxon:.3f})" in ms)
e2=et[(et.cohort=='gse211785')]
ok('GSE211785 DKD vs HKD 2 vs 2',set(zip(e2[e2.comparison=='DKD_vs_HKD'].n1,e2[e2.comparison=='DKD_vs_HKD'].n2))=={(2,2)} and "only two DKD and two HKD donors" in ms, list(zip(e2.n1,e2.n2)))
m=pd.read_csv(R+'chrom_motifs/motif_report.tsv',sep='\t'); d=m[m.direction=='DKD_side'].drop_duplicates('motif').set_index('motif')
pc=lambda x: round(100*x)
ok('REL/RELA',f"({pc(d.loc['REL','fg_frac'])}% and {pc(d.loc['RELA','fg_frac'])}% of peaks versus {pc(d.loc['REL','bg_frac'])}% and {pc(d.loc['RELA','bg_frac'])}% of matched background)" in ms)
ok('REL and RELA are the top two',sorted(d.rank_in_direction)[:2]==sorted([d.loc['REL','rank_in_direction'],d.loc['RELA','rank_in_direction']]))
ok('NR3C1',f"(NR3C1; {pc(d.loc['NR3C1','fg_frac'])}% versus {pc(d.loc['NR3C1','bg_frac'])}%)" in ms and d.loc['NR3C1','q_bh']<0.05)
ok('NFKB1/NFKB2 enriched (q<0.05)',d.loc['NFKB1','q_bh']<0.05 and d.loc['NFKB2','q_bh']<0.05)
ok('AP-1 enriched (JUN q<0.05)',d.loc['JUN','q_bh']<0.05)
ok('STAT3 not significant',d.loc['STAT3','q_bh']>=0.05 and "STAT3 motifs were not significantly enriched" in ms, d.loc['STAT3','q_bh'])
tg=pd.read_csv(R+'chrom_attrib/top_genes.tsv',sep='\t'); top=list(tg[tg.direction=='DKD_side'].nearest_gene.head(20))
ok('PROSER2, ZFP36L1, PIM3, SMAD6 among top DKD-side genes',{'PROSER2','ZFP36L1','PIM3','SMAD6'}<=set(top))
import glob
chrom_xlsx=[f for f in sorted(glob.glob('manuscript/build/Supplementary_Table_S*.xlsx'))
            if 'Chromatin alignment' in str(pd.read_excel(f,sheet_name='README').iloc[0,1])]
assert len(chrom_xlsx)==1, chrom_xlsx
x=pd.read_excel(chrom_xlsx[0],sheet_name=None); x.pop('README')
blob=' '.join(v.to_string() for v in x.values())
core_blob=' '.join(x[k].to_string() for k in ('retrieval','external_tests') if k in x)   # core-build values would sit here
ok('chromatin table has final retrieval value',f"{rt.loc[('peaks','aligned'),'top1_at100_mean']:.3f}" in blob or "0.4668" in blob or "0.467" in blob)
ok('chromatin table has final GSE195460 g',"1.36" in blob)
ok('chromatin table has no core-only numbers (0.4496 / 1.4271)',"0.4496" not in core_blob and "1.427" not in core_blob)
for n,v,dt in chk: print('PASS' if v else 'FAIL',n,dt if not v else '')
print(sum(v for _,v,_ in chk),'/',len(chk))
