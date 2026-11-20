import unittest
from unittest.mock import patch
import recall_flow as RF

def hit(i=1,**kw):
    return dict(ext_id='fact_'+str(i),content='地区导致账户被封，之前已讨论过',subject='tester',**kw)

class FullOnTests(unittest.TestCase):
    def test_need_false_is_observation_not_veto(self):
        p=[hit()];j={'need':False,'scores':[9]}
        self.assertEqual(RF.select(p,'地区问题',[],j)[0],[])
        self.assertEqual(RF.select(p,'地区问题',[],j,full_on=True)[0],p)
    def test_low_score_still_no_injection_even_explicit(self):
        p=[hit()];j={'need':False,'scores':[4]}
        self.assertEqual(RF.select(p,'你还记得地区问题吗',[],j,full_on=True,explicit=True)[0],[])
    def test_pin_requires_strict_relevance_even_with_direct_entity(self):
        p=[hit(from_pin=True,entity_hit=True,entity_direct=True)]
        e=[{'direct':True}]
        self.assertEqual(RF.select(p,'地区问题',e,{'need':True,'scores':[8]},full_on=True)[0],[])
        self.assertEqual(RF.select(p,'地区问题',e,{'need':False,'scores':[9]},full_on=True)[0],p)
    def test_unavailable_judge_does_not_inject_fullon(self):
        p=[hit(entity_hit=True,entity_direct=True)]
        self.assertEqual(RF.select(p,'地区问题',[{'direct':True}],None,full_on=True)[0],[])
    def test_pool_gate_no_rerank_and_no_rewrite(self):
        # 2026-11-12: full-on uses the cross-encoder for every non-empty pool; low score -> inject nothing, no LLM judge.
        with patch.object(RF.ENT,'match',return_value=[]),patch.object(RF,'build_pool',return_value=[hit(vec_sim=.2)]), \
             patch.object(RF,'relevance_probe',return_value={'fact_1':.2}):
            f=RF.recall(None,'焦虑的就是地区问题',full_on=True,use_rewrite=True,xscorer=lambda q,d:[0.05]*len(d),
                        rewriter=lambda *a:self.fail('rewrite called'),judge=lambda *a,**k:self.fail('rerank called'))
        self.assertEqual(f.selected,[])
        self.assertEqual(f.calls,{'rewrite':0,'rerank':0,'xrerank':1})
        self.assertIn('total_ms',f.timings)
    def test_fullon_has_broad_pool_and_compatibility_before_rerank(self):
        with patch.object(RF.ENT,'match',return_value=[]),patch.object(RF,'build_pool',return_value=[hit(vec_sim=.8)]) as pool, \
             patch.object(RF,'relevance_probe',return_value={'fact_1':.8}):
            f=RF.recall(None,'要是号又被封了咋办',full_on=True,compat=lambda q,c:True,
                        judge=lambda q,c,**kw:{'need':False,'scores':[9]})
        self.assertEqual(pool.call_args.kwargs['cap'],40)
        self.assertEqual(pool.call_args.kwargs['channel_cap'],80)
        self.assertEqual(pool.call_args.kwargs['vec_k'],24)
        self.assertTrue(callable(pool.call_args.kwargs['compat']))
        self.assertEqual(len(f.selected),1)
        self.assertFalse(f.model_need)
    def test_unrelated_pins_do_not_hog_or_enter_pool(self):
        p=[hit(i,vec_sim=.8) for i in range(40)]
        pin=hit(1000,user_pinned=1)
        def probe(con,text,pool):return {h['ext_id']:(.1 if h['ext_id']=='fact_1000' else .8) for h in pool}
        with patch.object(RF.ENT,'match',return_value=[]),patch.object(RF,'build_pool',return_value=p), \
             patch.object(RF,'relevance_probe',side_effect=probe):
            f=RF.recall(None,'地区问题',full_on=True,extra=[pin],judge=lambda q,c,**kw:{'need':False,'scores':[0]*len(c)})
        self.assertNotIn('fact_1000',[h['ext_id'] for h in f.pool])
        self.assertEqual(f.pool[0]['ext_id'],'fact_0')
    def test_duplicate_probe_keeps_vector_score_for_pin(self):
        import retrieve as R
        with patch.object(R,'by_rarity',return_value=[]):
            scores=RF.relevance_probe(None,'地区问题',[hit(vec_sim=.8),hit(from_pin=True)])
        self.assertEqual(scores['fact_1'],.8)
    def test_pin_does_not_buy_fullon_ranking_slots_and_provenance_restored(self):
        import hybrid, echo, sys, types
        normal=hit(1,status='confirmed',user_pinned=0,score=1)
        pin=hit(2,status='confirmed',user_pinned=1,score=1)
        fake=types.ModuleType('vectors');fake.search_many=lambda *a,**k:[]
        with patch.object(RF,'_terms',return_value=['地区']), \
             patch.object(hybrid,'_fts_any',return_value=[normal,pin]), \
             patch.object(hybrid,'_disowned',return_value=set()), \
             patch.object(echo,'echo_sources',return_value={}), \
             patch.dict(sys.modules,{'vectors':fake}):
            old=RF.build_pool(None,'地区',[],[],[])
            new=RF.build_pool(None,'地区',[],[],[],compat=lambda q,c:True)
        self.assertEqual(old[0]['ext_id'],'fact_2')
        self.assertEqual(new[0]['ext_id'],'fact_1')
        self.assertTrue(next(h for h in new if h['ext_id']=='fact_2')['user_pinned'])

    def test_hybrid_pin_cannot_bypass_actual_relevance(self):
        import hooks_engram_recall as H
        pin=hit(user_pinned=1)
        self.assertEqual(H.filter_pinned_by_relevance('地区问题',[pin],[pin],None),[])
        self.assertEqual(H.filter_pinned_by_relevance('地区问题',[pin],[pin],{'scores':[8]}),[])
        self.assertEqual(H.filter_pinned_by_relevance('地区问题',[pin],[pin],{'scores':[9]}),[pin])
        regular=hit(2,user_pinned=0)
        self.assertEqual(H.filter_pinned_by_relevance('地区问题',[regular],[regular],None),[regular])
    def test_hybrid_direct_pin_has_no_unjudged_fallback(self):
        p=[hit(user_pinned=1,entity_hit=True,entity_direct=True)]
        self.assertEqual(RF.select(p,'地区问题',[{'direct':True}],None)[0],[])
        self.assertEqual(RF.select(p,'地区问题',[{'direct':True}],{'need':True,'scores':[8]})[0],[])

    def test_compatibility_is_applied_to_all_fullon_selections(self):
        self.assertEqual(RF.select([hit()],'地区问题',[],{'need':True,'scores':[10]},
                                   full_on=True,compat=lambda q,c:False)[0],[])

if __name__=='__main__':unittest.main()
