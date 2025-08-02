import sqlite3
import numpy as np
from datetime import datetime, date
import threading
import os

class DatabaseManager:
    def __init__(self, db_path="data/hotel_recognition.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self.init_database()
        print(f"Database initialized at: {self.db_path}")
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Enable foreign keys
                cursor.execute("PRAGMA foreign_keys = ON")
                
                # Customers table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS customers (
                        customer_id TEXT PRIMARY KEY,
                        name TEXT,
                        embedding BLOB,
                        first_visit TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        total_visits INTEGER DEFAULT 0,
                        last_visit TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    )
                ''')
                
                # Staff table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS staff (
                        staff_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        department TEXT,
                        embedding BLOB,
                        added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    )
                ''')
                
                # Visits table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS visits (
                        visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        customer_id TEXT,
                        visit_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        confidence REAL,
                        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
                    )
                ''')
                
                # Staff detections table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS staff_detections (
                        detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        staff_id TEXT,
                        detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        confidence REAL,
                        FOREIGN KEY (staff_id) REFERENCES staff (staff_id)
                    )
                ''')
                
                # System logs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_logs (
                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        log_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        log_level TEXT,
                        message TEXT
                    )
                ''')
                # Add staff attendance table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS staff_attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        staff_id TEXT NOT NULL,
                        date DATE NOT NULL,
                        check_in_time TIME,
                        check_out_time TIME,
                        hours_worked REAL DEFAULT 0,
                        status TEXT DEFAULT 'Present',
                        recognition_confidence REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (staff_id) REFERENCES staff (staff_id),
                        UNIQUE(staff_id, date)
                    )
                ''')

                conn.commit()
                conn.close()
                print("Database tables created successfully")

        except Exception as e:
            print(f"Database initialization error: {e}")
            raise
    
    def register_new_customer(self, embedding, image=None):
        """Register a new customer"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Generate customer ID
                cursor.execute("SELECT COUNT(*) FROM customers")
                count = cursor.fetchone()[0]
                customer_id = f"CUST_{count + 1:06d}"
                
                # Store embedding as blob
                embedding_blob = embedding.tobytes() if embedding is not None else None
                
                cursor.execute('''
                    INSERT INTO customers (customer_id, embedding, total_visits)
                    VALUES (?, ?, 0)
                ''', (customer_id, embedding_blob))
                
                conn.commit()
                conn.close()
                
                print(f"New customer registered: {customer_id}")
                return customer_id
                
        except Exception as e:
            print(f"Error registering customer: {e}")
            return None
    
    def add_staff_member(self, staff_id, name, department, embedding, image=None):
        """Add a staff member"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                embedding_blob = embedding.tobytes() if embedding is not None else None
                
                cursor.execute('''
                    INSERT OR REPLACE INTO staff (staff_id, name, department, embedding)
                    VALUES (?, ?, ?, ?)
                ''', (staff_id, name, department, embedding_blob))
                
                conn.commit()
                conn.close()
                
                print(f"Staff member added: {staff_id} - {name}")
                return True
                
        except Exception as e:
            print(f"Error adding staff member: {e}")
            return False

    def record_visit(self, customer_id, confidence=1.0):
        """Record a customer visit with verification"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Verify customer exists first
                cursor.execute('SELECT customer_id FROM customers WHERE customer_id = ?', (customer_id,))
                if not cursor.fetchone():
                    print(f"❌ Cannot record visit: customer {customer_id} not found")
                    conn.close()
                    return False

                # Record visit
                cursor.execute('''
                    INSERT INTO visits (customer_id, confidence)
                    VALUES (?, ?)
                ''', (customer_id, confidence))

                visit_id = cursor.lastrowid

                # Update customer total visits
                cursor.execute('''
                    UPDATE customers
                    SET total_visits = total_visits + 1,
                        last_visit = CURRENT_TIMESTAMP
                    WHERE customer_id = ?
                ''', (customer_id,))

                conn.commit()
                conn.close()

                # Verify the visit was recorded
                if self._verify_visit_recorded(visit_id):
                    print(f"✅ Visit recorded and verified: {customer_id} (Visit ID: {visit_id})")
                    return True
                else:
                    print(f"❌ Visit verification failed: {customer_id}")
                    return False

        except Exception as e:
            print(f"❌ Error recording visit: {e}")
            return False

    def _verify_visit_recorded(self, visit_id):
        """Verify that a visit was actually recorded in the database"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute('SELECT visit_id FROM visits WHERE visit_id = ?', (visit_id,))
                result = cursor.fetchone()

                conn.close()
                return result is not None

        except Exception as e:
            print(f"Visit verification error: {e}")
            return False

    def record_staff_detection(self, staff_id, confidence=1.0):
        """Record a staff detection"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO staff_detections (staff_id, confidence)
                    VALUES (?, ?)
                ''', (staff_id, confidence))
                
                conn.commit()
                conn.close()
                
                return True
                
        except Exception as e:
            print(f"Error recording staff detection: {e}")
            return False
    
    def get_all_customers(self):
        """Get all customers"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT customer_id, name, embedding, first_visit, total_visits, last_visit
                    FROM customers WHERE is_active = 1
                ''')
                
                customers = []
                for row in cursor.fetchall():
                    customers.append({
                        'customer_id': row[0],
                        'name': row[1],
                        'embedding': row[2],
                        'first_visit': row[3],
                        'total_visits': row[4],
                        'last_visit': row[5]
                    })
                
                conn.close()
                return customers
                
        except Exception as e:
            print(f"Error getting customers: {e}")
            return []

    def load_customers(self):
        """Load all active customers and their embeddings"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT customer_id, embedding FROM customers WHERE is_active = 1 AND embedding IS NOT NULL")
                
                customers = []
                for row in cursor.fetchall():
                    customer_id, embedding_blob = row
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    customers.append({'id': customer_id, 'embedding': embedding})
                
                conn.close()
                return customers
                
        except Exception as e:
            print(f"Error loading customers: {e}")
            return []
    
    def get_all_staff(self):
        """Get all staff members"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT staff_id, name, department, embedding, added_date
                    FROM staff WHERE is_active = 1
                ''')
                
                staff_members = []
                for row in cursor.fetchall():
                    staff_members.append({
                        'staff_id': row[0],
                        'name': row[1],
                        'department': row[2],
                        'embedding': row[3],
                        'added_date': row[4]
                    })
                
                conn.close()
                return staff_members
                
        except Exception as e:
            print(f"Error getting staff: {e}")
            return []
    
    def get_customer_info(self, customer_id):
        """Get customer information"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT customer_id, name, total_visits, last_visit
                    FROM customers WHERE customer_id = ?
                ''', (customer_id,))
                
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    return {
                        'customer_id': row[0],
                        'name': row[1],
                        'total_visits': row[2],
                        'last_visit': row[3]
                    }
                
                return None
                
        except Exception as e:
            print(f"Error getting customer info: {e}")
            return None
    
    def get_staff_info(self, staff_id):
        """Get staff information"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT staff_id, name, department
                    FROM staff WHERE staff_id = ?
                ''', (staff_id,))
                
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    return {
                        'staff_id': row[0],
                        'name': row[1],
                        'department': row[2]
                    }
                
                return None
                
        except Exception as e:
            print(f"Error getting staff info: {e}")
            return None
    
    def is_new_visit_today(self, customer_id):
        """Check if this is a new visit today"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT COUNT(*) FROM visits 
                    WHERE customer_id = ? AND DATE(visit_time) = DATE('now')
                ''', (customer_id,))
                
                count = cursor.fetchone()[0]
                conn.close()
                
                return count == 0
                
        except Exception as e:
            print(f"Error checking visit: {e}")
            return True

    def get_daily_visits(self, date):
        """Get all visits for a specific date"""
        try:
            query = """
            SELECT v.*, c.customer_id, c.name 
            FROM visits v
            LEFT JOIN customers c ON v.customer_id = c.customer_id
            WHERE DATE(v.timestamp) = ?
            ORDER BY v.timestamp DESC
            """

            result = self.execute_query(query, (date.strftime('%Y-%m-%d'),), fetch=True)

            visits = []
            for row in result:
                visits.append({
                    'visit_id': row[0],
                    'customer_id': row[1],
                    'timestamp': row[2],
                    'confidence': row[3],
                    'name': row[5] if len(row) > 5 else None
                })

            return visits

        except Exception as e:
            print(f"Error getting daily visits: {e}")
            return []

    def get_customer_visits(self, customer_id):
        """Get all visits for a specific customer"""
        try:
            query = """
            SELECT * FROM visits 
            WHERE customer_id = ?
            ORDER BY timestamp DESC
            """

            result = self.execute_query(query, (customer_id,), fetch=True)
            return result if result else []

        except Exception as e:
            print(f"Error getting customer visits: {e}")
            return []

    def get_daily_staff_detections(self, target_date=None):
        """Get daily staff detections"""
        if target_date is None:
            target_date = date.today()
        
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT sd.detection_time, sd.staff_id, s.name, s.department
                    FROM staff_detections sd
                    JOIN staff s ON sd.staff_id = s.staff_id
                    WHERE DATE(sd.detection_time) = ?
                    ORDER BY sd.detection_time
                ''', (target_date,))
                
                detections = []
                for row in cursor.fetchall():
                    detections.append({
                        'detection_time': datetime.fromisoformat(row[0]),
                        'staff_id': row[1],
                        'staff_name': row[2],
                        'department': row[3]
                    })
                
                conn.close()
                return detections
                
        except Exception as e:
            print(f"Error getting staff detections: {e}")
            return []
    
    def get_monthly_statistics(self, year, month):
        """Get monthly statistics"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Total visits in month
                cursor.execute('''
                    SELECT COUNT(*) FROM visits 
                    WHERE strftime('%Y', visit_time) = ? AND strftime('%m', visit_time) = ?
                ''', (str(year), f"{month:02d}"))
                total_visits = cursor.fetchone()[0]
                
                # Unique customers in month
                cursor.execute('''
                    SELECT COUNT(DISTINCT customer_id) FROM visits 
                    WHERE strftime('%Y', visit_time) = ? AND strftime('%m', visit_time) = ?
                ''', (str(year), f"{month:02d}"))
                unique_customers = cursor.fetchone()[0]
                
                # New customers in month
                cursor.execute('''
                    SELECT COUNT(*) FROM customers 
                    WHERE strftime('%Y', first_visit) = ? AND strftime('%m', first_visit) = ?
                ''', (str(year), f"{month:02d}"))
                new_customers = cursor.fetchone()[0]
                
                conn.close()
                
                return {
                    'total_visits': total_visits,
                    'unique_customers': unique_customers,
                    'new_customers': new_customers,
                    'avg_visits_per_day': total_visits / 30.0,
                    'daily_breakdown': []
                }
                
        except Exception as e:
            print(f"Error getting monthly statistics: {e}")
            return {
                'total_visits': 0,
                'unique_customers': 0,
                'new_customers': 0,
                'avg_visits_per_day': 0.0,
                'daily_breakdown': []
            }
    
    def test_database_connection(self):
        """Test database connection and tables"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Check if tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                print(f"Database tables: {[table[0] for table in tables]}")
                
                # Check customer count
                cursor.execute("SELECT COUNT(*) FROM customers")
                customer_count = cursor.fetchone()[0]
                
                # Check staff count
                cursor.execute("SELECT COUNT(*) FROM staff")
                staff_count = cursor.fetchone()[0]
                
                print(f"Customers: {customer_count}, Staff: {staff_count}")
                
                conn.close()
                return True
                
        except Exception as e:
            print(f"Database test failed: {e}")
            return False
# Add these compatibility methods at the end of your DatabaseManager class
    def create_staff_attendance_table(self):
        """Create staff attendance tracking table"""
        query = """
        CREATE TABLE IF NOT EXISTS staff_attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            staff_id TEXT NOT NULL,
            date DATE NOT NULL,
            check_in_time TIME,
            check_out_time TIME,
            hours_worked REAL DEFAULT 0,
            status TEXT DEFAULT 'Present',
            recognition_confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (staff_id) REFERENCES staff_members (staff_id),
            UNIQUE(staff_id, date)
        )
        """
        self.execute_query(query)


    def record_staff_attendance(self, staff_id, attendance_type='check_in', confidence=1.0):
        """Record staff check-in or check-out and return status information"""
        try:
            current_date = date.today().strftime('%Y-%m-%d')
            current_time = datetime.now().strftime('%H:%M:%S')

            already_checked_in = False

            if attendance_type == 'check_in':
                # Check if already checked in today
                existing = self.execute_query(
                    "SELECT id FROM staff_attendance WHERE staff_id = ? AND date = ?",
                    (staff_id, current_date), fetch=True
                )

                if existing:
                    already_checked_in = True
                    # Update existing record
                    query = """
                    UPDATE staff_attendance
                    SET check_in_time = ?, recognition_confidence = ?
                    WHERE staff_id = ? AND date = ?
                    """
                    self.execute_query(query, (current_time, confidence, staff_id, current_date))
                else:
                    # Insert new record
                    status = 'Late' if datetime.now().time() > datetime.strptime('09:00:00',
                                                                                 '%H:%M:%S').time() else 'Present'
                    query = """
                    INSERT INTO staff_attendance (staff_id, date, check_in_time, status, recognition_confidence)
                    VALUES (?, ?, ?, ?, ?)
                    """
                    self.execute_query(query, (staff_id, current_date, current_time, status, confidence))

            elif attendance_type == 'check_out':
                # Update check-out time and calculate hours
                query = """
                UPDATE staff_attendance
                SET check_out_time = ?,
                    hours_worked = CASE
                        WHEN check_in_time IS NOT NULL THEN
                            (julianday(date || ' ' || ?) - julianday(date || ' ' || check_in_time)) * 24
                        ELSE 0
                    END
                WHERE staff_id = ? AND date = ?
                """
                self.execute_query(query, (current_time, current_time, staff_id, current_date))

            # Get total visits for staff member
            total_visits_result = self.execute_query(
                "SELECT COUNT(*) FROM staff_attendance WHERE staff_id = ?",
                (staff_id,), fetch=True
            )
            total_visits = total_visits_result[0][0] if total_visits_result else 0

            return {
                'success': True,
                'already_checked_in': already_checked_in,
                'total_visits': total_visits
            }

        except Exception as e:
            print(f"Error recording staff attendance: {e}")
            return {'success': False, 'already_checked_in': False, 'total_visits': 0}


    def get_staff_attendance_report(self, start_date, end_date):
        """Get staff attendance report for date range"""
        query = """
        SELECT 
            sm.staff_id,
            sm.name,
            sm.department,
            COUNT(sa.date) as total_days,
            SUM(CASE WHEN sa.status IN ('Present', 'Late') THEN 1 ELSE 0 END) as present_days,
            SUM(CASE WHEN sa.status = 'Absent' THEN 1 ELSE 0 END) as absent_days,
            SUM(CASE WHEN sa.status = 'Late' THEN 1 ELSE 0 END) as late_days,
            AVG(sa.hours_worked) as avg_hours_per_day,
            ROUND(
                (SUM(CASE WHEN sa.status IN ('Present', 'Late') THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2
            ) as attendance_percentage
        FROM staff_members sm
        LEFT JOIN staff_attendance sa ON sm.staff_id = sa.staff_id 
            AND sa.date BETWEEN ? AND ?
        GROUP BY sm.staff_id, sm.name, sm.department
        ORDER BY attendance_percentage DESC
        """
        return self.execute_query(query, (start_date, end_date), fetch=True)


    def load_customers(self):
        """Compatibility alias for get_all_customers"""
        print("🔄 Loading customers via alias method...")
        return self.get_all_customers()

    def execute_query(self, query, params=None, fetch=False):
        """Execute SQL query with proper error handling"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                if fetch:
                    result = cursor.fetchall()
                    conn.close()
                    return result
                else:
                    conn.commit()
                    conn.close()
                    return True

        except Exception as e:
            print(f"Database query error: {e}")
            if 'conn' in locals():
                conn.close()
            return False if not fetch else []

    def get_database_stats(self):
        """Get current database statistics"""
        try:
            stats = {}
            tables = ['customers', 'visits', 'staff_detections', 'staff', 'staff_attendance']

            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                for table in tables:
                    try:
                        cursor.execute(f'SELECT COUNT(*) FROM {table}')
                        count = cursor.fetchone()[0]
                        stats[table] = count
                    except sqlite3.OperationalError:
                        stats[table] = 'Table not found'

                conn.close()
                return stats

        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}

    def delete_staff_member(self, staff_id):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM staff WHERE staff_id = ?", (staff_id,))
            conn.commit()
            conn.close()
            return True
        except:
            return False

    def reset_recognition_data(self):
        """Reset all recognition data while keeping system structure"""
        try:
            # Create backup first
            import shutil
            backup_path = f"{self.db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(self.db_path, backup_path)
            print(f"✅ Backup created: {backup_path}")

            # Tables to reset (keep structure, clear data)
            tables_to_reset = [
                'customers',
                'visits',
                'staff_detections',
                'staff_attendance'
            ]

            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Disable foreign key constraints temporarily
                cursor.execute('PRAGMA foreign_keys = OFF')

                for table in tables_to_reset:
                    try:
                        # Clear all data
                        cursor.execute(f'DELETE FROM {table}')

                        # Reset auto-increment
                        cursor.execute(f'DELETE FROM sqlite_sequence WHERE name = ?', (table,))

                        print(f"✅ Reset table: {table}")

                    except sqlite3.OperationalError as e:
                        if "no such table" not in str(e).lower():
                            print(f"⚠️  Could not reset {table}: {e}")

                # Re-enable foreign key constraints
                cursor.execute('PRAGMA foreign_keys = ON')

                conn.commit()
                conn.close()

            print("🎉 Recognition data reset successfully!")
            return True

        except Exception as e:
            print(f"❌ Database reset failed: {e}")
            return False

    def load_staff(self):
        """Load all active staff and their embeddings"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute("SELECT staff_id, embedding FROM staff WHERE is_active = 1 AND embedding IS NOT NULL")

                staff = []
                for row in cursor.fetchall():
                    staff_id, embedding_blob = row
                    if embedding_blob:
                        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                        staff.append({'id': staff_id, 'embedding': embedding})

                conn.close()
                return staff

        except Exception as e:
            print(f"Error loading staff: {e}")
            return []


# Test the database manager if run directly
if __name__ == "__main__":
    db = DatabaseManager()
    db.test_database_connection()
