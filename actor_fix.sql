-- 说话人标注修正层（overlay）：evidence 表只增不改，历史上标错的说话人在这里改判，原行一字不动。
-- 一行 = 一条被改判的 evidence。读的人（抽取检查/蒸馏渲染/防回声/回放）先问这张表，没有记录才用 evidence.source_actor。
--   leading-block ：整条正文就是 <channel user=… user_id=…> 转发包装，说话人以包装为准（例：bot 转发被标成她的话）。
--   embedded-block：助手那条证据里夹着另一个人的 <channel> 块：前半仍是原说话人，块里的话归 block_actor，
--                   但只有「同一会话里另有一条该说话人的证据含同样的话」（corroborated=1）才算数；
--                   对不上的不当任何人的话（助手侧 channel 块常是模型续写出的假「下一轮」）。
CREATE TABLE IF NOT EXISTS evidence_actor_fix (
  evidence_id     INTEGER PRIMARY KEY REFERENCES evidence (id),
  kind            TEXT NOT NULL CHECK (kind IN ('leading-block', 'embedded-block')),
  original_actor  TEXT,
  effective_actor TEXT,          -- leading-block：改判后的说话人（认不出为 NULL）；embedded-block：NULL
  block_actor     TEXT,          -- embedded-block：块里那个人
  block_user      TEXT,          -- 块的 user 属性（只存名字/数字 id，不存正文）
  corroborated    INTEGER,       -- embedded-block：1＝另有原话佐证，0＝没有
  reason          TEXT NOT NULL,
  created_at      TEXT NOT NULL
);
CREATE VIEW IF NOT EXISTS evidence_effective AS
  SELECT e.id AS evidence_id, e.source_actor AS original_actor,
         CASE WHEN f.kind = 'leading-block' THEN COALESCE(f.effective_actor, 'unknown') ELSE e.source_actor END AS effective_actor,
         f.kind AS fix_kind, f.block_actor, f.corroborated, f.reason AS fix_reason
  FROM evidence e LEFT JOIN evidence_actor_fix f ON f.evidence_id = e.id;
