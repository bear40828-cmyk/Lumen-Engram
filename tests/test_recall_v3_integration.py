import io,json,unittest
from unittest.mock import patch,MagicMock
import hooks_engram_recall as hook
import flags
from retrieve import Result,log_recall

class BranchFixesTest(unittest.TestCase):
 def test_old_route_sees_clean_user_turn(self):
  data={'prompt':'【测试当前联系人资料｜元数据】\n文件已过时\n\nLumen页面需要哪些功能？'}
  seen=[]
  def skip(t):seen.append(t);return 'test exit'
  with patch('sys.stdin',io.StringIO(json.dumps(data))),patch.object(flags,'on',return_value=False),patch.object(flags,'raw',return_value=None),patch.object(hook,'pre_recall_skip',side_effect=skip),patch.object(hook,'log'):
   self.assertEqual(hook.main(),0)
  self.assertEqual(seen,['Lumen页面需要哪些功能？'])
 def test_skip_does_not_claim_context_selection(self):
  con=MagicMock();con.execute.return_value.fetchone.return_value=[1]
  run=MagicMock();run.run_id=None
  cm=MagicMock();cm.__enter__.return_value=run
  r=Result(True,hits=[{'ext_id':'fact_1','content':'x'}])
  with patch('telemetry.Run',return_value=cm):log_recall(con,'x',r,decision='skip',reason='off_topic')
  self.assertFalse(run.add_hit.call_args.kwargs['selected'])
 def test_explicit_recall_selection_remains_exact(self):
  con=MagicMock();con.execute.return_value.fetchone.return_value=[1]
  run=MagicMock();run.run_id=None
  cm=MagicMock();cm.__enter__.return_value=run
  r=Result(True,hits=[{'ext_id':'fact_1','content':'x'}])
  with patch('telemetry.Run',return_value=cm):log_recall(con,'x',r,decision='recall',selected={'fact_1'})
  self.assertTrue(run.add_hit.call_args.kwargs['selected'])
if __name__=='__main__':unittest.main()


class FrontFilterTest(unittest.TestCase):
    def test_task_heads_and_app_markers_are_not_user_turns(self):
        import recall_flow as RF
        for t in ('【自主唤醒入口｜服务器任务，不是用户消息】读取文件',
                  '【定时体检｜不是用户消息】跑一遍检查',
                  '[[bear-pat|5oiR|562J|IOea]]'):
            self.assertIsNone(RF.clean_message(t), t)

    def test_plain_turns_survive(self):
        import recall_flow as RF
        for t in ('要是号又被封了怎么办', '【重点】明天几点出发', '看这个 [[wiki|链接]] 写得对吗'):
            self.assertEqual(RF.clean_message(t), t)

    def test_latin_keywords_not_truncated(self):
        from router import keywords
        ks = keywords('这个 processed 字段改成 true 了吗', k=4)
        self.assertIn('processed', ks)
        self.assertNotIn('processe', ks)
