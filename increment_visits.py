# increment_visits.py - Testing visit count increment
import sqlite3
from datetime import datetime, date

def increment_customer_visits(customer_id="CUST_000003"):
    """Increment visit count for testing"""
    try:
        db_path = "data/hotel_recognition.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        today = date.today()
        
        # Get current visit counts
        cursor.execute('SELECT total_visits FROM customers WHERE customer_id = ?', (customer_id,))
        result = cursor.fetchone()
        
        if result:
            current_total = result[0]
            new_total = current_total + 1
            
            print(f"📊 Current visits: {current_total}")
            print(f"🔄 Updating to: {new_total}")
            
            # Update customer total visits
            cursor.execute('''
                UPDATE customers 
                SET total_visits = ?, last_visit = ? 
                WHERE customer_id = ?
            ''', (new_total, datetime.now(), customer_id))
            
            # Update daily summary if exists
            cursor.execute('''
                UPDATE daily_visit_summary 
                SET total_visits_today = total_visits_today + 1,
                    total_visits_overall = ?
                WHERE customer_id = ? AND visit_date = ?
            ''', (new_total, customer_id, today))
            
            # Insert new visit record
            cursor.execute('''
                INSERT OR IGNORE INTO visits (customer_id, visit_date, visit_time, confidence)
                VALUES (?, ?, ?, 0.9)
            ''', (customer_id, today, datetime.now()))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Visit count updated successfully!")
            print(f"🎯 {customer_id} now has {new_total} total visits")
            return True
        else:
            print(f"❌ Customer {customer_id} not found")
            return False
            
    except Exception as e:
        print(f"❌ Error updating visits: {e}")
        return False

if __name__ == "__main__":
    increment_customer_visits()
