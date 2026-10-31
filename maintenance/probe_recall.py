"""召回自检（不需要模型）：随机抽 150 条活事实，用事实自己的关键词当问句，看它能不能被召回。
用法：python3 maintenance/probe_recall.py <库路径>。开/关某个功能前后各跑一次比数字。"""
import sys,os,sqlite3,random; ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
from store.store import connect
from retrieve import retrieve
from router import keywords
db=sys.argv[1]
con=connect(db)
random.seed(11)
ids=[r[0] for r in sqlite3.connect(db, uri=False).execute("select ext_id from facts where status='candidate' and valid_to is null and fact_type!='semantic' and length(content)>25 order by id")]
sample=random.sample(ids,150)
hit=0;top3=0;ranks=[]
for e in sample:
    c=con.execute("select content from facts where ext_id=?",(e,)).fetchone()[0]
    q=" ".join(keywords(c,k=3))
    r=retrieve(con,q)
    exts=[h['ext_id'] for h in r.hits]
    if e in exts:
        hit+=1; k=exts.index(e); ranks.append(k)
        if k<3: top3+=1
print(db,'hit@all',hit,'top3',top3,'of',len(sample),'meanrank',round(sum(ranks)/max(1,len(ranks)),2))
for q in ["用户考研备考科目","用户养的宠物是什么","备考科目 301","备考科目 302"]:
    r=retrieve(con,q)
    print(' Q',q,[ (i,h['content'][:18]) for i,h in enumerate(r.hits[:6]) if '巩固' in h['content']])
