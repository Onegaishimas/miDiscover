import psycopg2

try:
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="mechinterp_discovery",
        user="mechinterp",
        password="mechinterp_dev_password"
    )
    cur = conn.cursor()
    
    # Check schemas
    cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'discovery';")
    result = cur.fetchone()
    
    if result:
        print("✅ Discovery schema exists!")
        
        # Check tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'discovery'
            ORDER BY table_name;
        """)
        tables = cur.fetchall()
        
        if tables:
            print("📋 Tables in discovery schema:")
            for table in tables:
                print(f"   - {table[0]}")
        else:
            print("❌ No tables found in discovery schema")
    else:
        print("❌ Discovery schema not found")
    
    cur.close()
    conn.close()
    
except Exception as e:
    print(f"❌ Connection error: {e}")
