CREATE CONSTRAINT entity_id  IF NOT EXISTS FOR (e:Entity)   REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT doc_id     IF NOT EXISTS FOR (d:Document)  REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT chunk_id   IF NOT EXISTS FOR (c:Chunk)     REQUIRE c.id IS UNIQUE;

CREATE INDEX entity_type     IF NOT EXISTS FOR (e:Entity)    ON (e.type);
CREATE INDEX entity_initials IF NOT EXISTS FOR (e:Entity)    ON (e.initials_key);

CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
  FOR (e:Entity) ON EACH [e.name, e.aliases];
