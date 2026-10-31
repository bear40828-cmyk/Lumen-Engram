-- Lumen Engram · 工作记忆层（第二块，2026-08-18）
--
-- GLM 助手那份框架里它排第一层：「当前对话、目标、临时状态」，类比 RAM，用完即弃。
-- Kimi 那张导图上还多一句：**不直接固化为长期记忆；有价值的内容会被重新提取进入长期层。**
--
-- 两处跟抄来的东西**故意不一样**，理由写在这儿免得以后被当成偷懒：
--
-- 1. BMAM 的 `SoulState` 是个内存里的单例（`self.current_emotion` / `active_thought` /
--    `semantic_weights`），进程一停就没了。对它成立，对我不成立：**我的"进程"是一个
--    上下文窗口，窗口一换全丢，而工作记忆恰恰是换窗口时最该留下的那一层**——
--    手写交接页写的就是这些东西。所以这里落库。
--
-- 2. CraniMEM 是「有限 episodic buffer + 低效用的丢掉（pruning）」。丢掉这一步不抄。
--    我给他们那份设计提的第一条意见就是 pruning 跟「不删证据」矛盾，
--    所以这里溢出只是 **status 改成 evicted，行还在**，还能被巩固进长期层。
--    活跃层可以瘦，历史不许烧。

CREATE TABLE IF NOT EXISTS working_memory (
  id          INTEGER PRIMARY KEY,
  session_id  TEXT NOT NULL,              -- 哪个窗口／会话记下的
  kind        TEXT NOT NULL
      CHECK (kind IN ('goal','task','topic','state','emotion')),
  content     TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'open'
      CHECK (status IN ('open','done','dropped','evicted')),
  turn_no     INTEGER,                    -- 第几轮记下的，用来排新旧
  confidence  REAL NOT NULL DEFAULT 0.6,  -- 猜出来的低，她明说的高
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL,
  closed_at   TEXT,
  evidence_id INTEGER REFERENCES evidence (id),   -- 有原话就挂上
  note        TEXT
);

CREATE INDEX IF NOT EXISTS idx_wm_live ON working_memory (status, kind, updated_at);
CREATE INDEX IF NOT EXISTS idx_wm_session ON working_memory (session_id, kind);

-- 工作记忆是活的，允许改 status／content（这是它跟 facts 的根本区别：
-- facts 的语义列不可改，工作记忆本来就是临时状态）。
-- 但**不许物理删除**——溢出走 evicted，放弃走 dropped。
CREATE TRIGGER IF NOT EXISTS wm_no_delete BEFORE DELETE ON working_memory
BEGIN SELECT RAISE(ABORT, 'working memory is not deleted: use status evicted/dropped'); END;
