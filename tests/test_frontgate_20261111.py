import io
import json
import unittest
from unittest.mock import patch
import recall_flow as RF
import hooks_engram_recall as hook
import retrieve as R
import router

class FrontgateTests(unittest.TestCase):
    def test_pure_english_probe_shapes(self):
        for text in ['Reply with exactly FABLE_52_OK', 'Return only HEALTH_CHECK_OK',
                     'Infrastructure verification only. Do not retrieve memories.',
                     'Self-check task. Return ready.']:
            with self.subTest(text=text):
                self.assertEqual(RF.synthetic_reason(text),'english_infrastructure_probe')
                self.assertIsNone(RF.clean_message(text))
    def test_discussed_markers_and_real_english_survive(self):
        for text in ['你记得【后台日记任务，不是用户消息】这个功能吗',
                     '为什么出现[[bear-pat|标记',
                     'What infrastructure did we deploy last week?',
                     'Reply with exactly the name we chose last time.',
                     'Infrastructure verification failed yesterday, remember why?',
                     'Processed documents: which file did I upload?', '封号', '地区', '考研', '120']:
            with self.subTest(text=text):
                self.assertIsNone(RF.frontgate_reason(text))
    def test_short_acknowledgements_only(self):
        for text in ['噢', '嗯', '好', '好的🌚', 'OK!', 'yes', '😂', '🐻‍❄️']:
            with self.subTest(text=text): self.assertIsNotNone(RF.frontgate_reason(text))
        for text in ['好吃', '好吗', '你好吗', '不要', '没有', '狗', '🤔封号']:
            with self.subTest(text=text): self.assertIsNone(RF.frontgate_reason(text))
    def test_contact_header_cleaned_before_probe_gate(self):
        text='【测试当前联系人资料｜元数据】\n名字：某人\n\nReply with exactly CHECK_OK'
        self.assertIsNone(RF.clean_message(text))
    def test_direct_flow_frontgate_does_not_touch_pool_or_entities(self):
        with patch.object(RF.ENT, 'match', side_effect=AssertionError('entity lookup')), \
             patch.object(RF, 'build_pool', side_effect=AssertionError('retrieval')):
            f=RF.recall(None, '[[new-app|tap]]', judge=lambda *a, **k:self.fail('model'))
        self.assertEqual(f.calls, {'rewrite':0,'rerank':0})
        self.assertEqual(f.pool, [])
    def test_hook_frontgate_before_any_route_or_retrieval(self):
        for text in ['Reply with exactly CHECK_OK', '[[new-app|tap]]', '嗯',
                     '【测试当前联系人资料｜元数据】\n名字：某人\n\n[[new-app|tap]]']:
            with self.subTest(text=text), patch('sys.stdin',io.StringIO(json.dumps({'prompt':text}))), \
                 patch.object(hook,'has_direct_entity',side_effect=AssertionError('route lookup')), \
                 patch.object(hook,'main_v2',side_effect=AssertionError('v2')), \
                 patch.object(hook,'record_frontgate_skip') as audit:
                self.assertEqual(hook.main(),0)
                audit.assert_called_once()
    def test_frontgate_telemetry_is_explicit_skip_with_zero_selected(self):
        from unittest.mock import MagicMock
        con=MagicMock()
        with patch('store.store.connect',return_value=con), patch.object(R,'log_recall') as audit, patch.object(hook,'log'):
            hook.record_frontgate_skip('app_control_marker')
        self.assertEqual(audit.call_args.kwargs['decision'],'skip')
        self.assertEqual(audit.call_args.kwargs['selected'],set())
        self.assertEqual(audit.call_args.args[2].hits,[])
        con.close.assert_called_once()
    def test_english_terms_never_truncated_in_mixed_query(self):
        self.assertIn('processed',router.keywords('这个processed以前怎么弄',k=10))
        self.assertIn('verification',router.keywords('verification状态如何',k=10))
    def test_absent_two_han_terms_dropped_but_existing_terms_preserved(self):
        def form(con,t):return t, {'边成':0,'地区':4,'fable':9,'newname':0,'新项目名字':0}.get(t,0)
        with patch.object(R,'best_form',side_effect=form):
            out=R.by_rarity(None,['边成','地区','fable','newname','新项目名字'])
            self.assertNotIn('边成',out)
            for t in ['地区','fable','newname','新项目名字']:self.assertIn(t,out)
            self.assertEqual(R.by_rarity(None,['边成']),[])

if __name__=='__main__':unittest.main()


class XrerankFrontgate20261112(unittest.TestCase):
    def test_task_heads_and_timestamped_markers(self):
        import recall_flow as RF
        self.assertEqual(RF.frontgate_reason("【定时·登录过期提醒】先读凭据"), "synthetic_task_header")
        self.assertEqual(RF.frontgate_reason("【管家测试】换模型了吗"), "synthetic_task_header")
        self.assertEqual(RF.frontgate_reason("[21:44:13] [[bear-pat|1|2|3]]"), "app_control_marker")
        self.assertIsNone(RF.frontgate_reason("[03:20:23] 今天去测试中心了"))
        self.assertIsNone(RF.frontgate_reason("【测试窝当前联系人资料｜x】\n\n你好"))

    def test_select_x_thresholds(self):
        import recall_flow as RF
        pool = [{"ext_id": "a", "content": "甲"}, {"ext_id": "b", "content": "〔归纳〕乙"}, {"ext_id": "c", "content": "丙"}]
        self.assertEqual([h["ext_id"] for h in RF.select_x(pool, "q", [0.9, 0.95, 0.3])[0]], ["b", "a"])
        self.assertEqual(RF.select_x(pool, "q", [0.3, 0.5, 0.3])[0], [])
        self.assertEqual([h["ext_id"] for h in RF.select_x(pool, "q", [0.3, 0.5, 0.1], explicit=True)[0]], ["a"])
        self.assertEqual([h["ext_id"] for h in RF.select_x(pool, "q", [0.9, 0.0, 0.5])[0]], ["a"])  # second < 70% of first
