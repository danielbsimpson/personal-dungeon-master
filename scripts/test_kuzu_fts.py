"""Quick test: Kuzu 0.11.3 FTS index creation syntax."""
import kuzu

db = kuzu.Database(":memory:")
conn = kuzu.Connection(db)

conn.execute("INSTALL fts")
conn.execute("LOAD EXTENSION fts")
conn.execute("CREATE NODE TABLE IF NOT EXISTS TestTbl (uuid STRING PRIMARY KEY, name STRING, summary STRING)")
conn.execute("CALL CREATE_FTS_INDEX('TestTbl', 'test_idx', ['name', 'summary'])")
print("FTS index creation OK")

# Test duplicate creation
try:
    conn.execute("CALL CREATE_FTS_INDEX('TestTbl', 'test_idx', ['name', 'summary'])")
    print("Duplicate: allowed")
except Exception as e:
    print("Duplicate raises:", type(e).__name__, str(e)[:80])

# Test querying it
conn.execute("CREATE (:TestTbl {uuid: '1', name: 'Mira Coldstone', summary: 'A mysterious wizard'})")
results = conn.execute("CALL QUERY_FTS_INDEX('TestTbl', 'test_idx', 'Mira', TOP := 5) RETURN node.name")
print("FTS query results:", list(results))
