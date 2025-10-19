import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime

# Path to your XML file
xml_file = r"C:\Users\siddhartha\gmail_llama_pipeline\data\raw\sms\sms-20251017013727.xml"

# Parse XML
tree = ET.parse(xml_file)
root = tree.getroot()

# Get all SMS elements
all_sms = list(root.findall('sms'))
total_messages = len(all_sms)

print(f"📊 Total messages in XML: {total_messages}")
print("\n🔍 Showing LAST 3 messages from XML (these will be FIRST in CSV):")

# Show last 3 messages (bottom of XML)
for i in range(total_messages - 1, max(total_messages - 4, -1), -1):
    sms = all_sms[i]
    sender = sms.get('address', '')
    timestamp = sms.get('date', '')
    body = sms.get('body', '')[:50]
    try:
        ts_readable = datetime.fromtimestamp(int(timestamp)/1000).strftime('%Y-%m-%d %H:%M:%S')
    except:
        ts_readable = timestamp
    print(f"  XML Position {i+1}:")
    print(f"    Sender: {sender}")
    print(f"    Time: {ts_readable}")
    print(f"    Body preview: {body}...")
    print()

# Ask user how many rows to convert
user_input = input(f"How many rows to convert FROM BOTTOM (reverse order)? (press Enter for all {total_messages}): ").strip()

if user_input == "":
    max_rows = None
    print(f"\n✅ Converting ALL {total_messages} messages from BOTTOM in REVERSE order...")
else:
    try:
        max_rows = int(user_input)
        print(f"\n✅ Converting LAST {max_rows} messages from BOTTOM in REVERSE order...")
    except ValueError:
        print("\n⚠️  Invalid input. Converting all messages...")
        max_rows = None

# Calculate which messages to process
if max_rows:
    messages_to_process = min(max_rows, total_messages)
else:
    messages_to_process = total_messages

# Start from the LAST message and go backwards
start_index = total_messages - 1  # Last message
end_index = total_messages - messages_to_process  # Stop here (exclusive)

print(f"📍 Processing XML positions from {start_index + 1} down to {end_index + 1}")
print(f"   (Converting {messages_to_process} messages from bottom in reverse)\n")

# Process data - FROM BOTTOM TO TOP (REVERSE ORDER)
data = []
csv_row_number = 1

# Iterate backwards: from last to first
for idx in range(start_index, end_index - 1, -1):
    sms = all_sms[idx]
    
    sender = sms.get('address', '')
    body = sms.get('body', '').replace('\n', ' ').strip()
    timestamp = sms.get('date', '')
    
    # Parse timestamp
    try:
        timestamp_readable = datetime.fromtimestamp(int(timestamp)/1000).strftime('%Y-%m-%d %H:%M:%S')
    except:
        timestamp_readable = timestamp
    
    data.append({
        "csv_row": csv_row_number,
        "xml_position": idx + 1,
        "id": f"sms_{csv_row_number}",
        "source": "sms",
        "sender": sender,
        "subject": "(none)",
        "body": body,
        "timestamp": timestamp_readable,
        "timestamp_ms": timestamp,
        "category": "",
        "priority_label": "",
        "action_required": "",
        "action_text": ""
    })
    
    csv_row_number += 1
    
    # Progress indicator
    if csv_row_number % 100 == 0:
        print(f"  Processed {csv_row_number}/{messages_to_process} messages...")

# Convert to DataFrame
df = pd.DataFrame(data)

# Save CSV
output_file = "sms_parsed.csv"
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✅ Saved {len(df)} messages to {output_file}")
print(f"📋 Order: XML position {start_index + 1} → {end_index + 1} (reverse)")

if max_rows and max_rows < total_messages:
    skipped_from_top = total_messages - max_rows
    print(f"ℹ️  Skipped {skipped_from_top} messages from the TOP of XML")

# Show summary
print("\n🔍 First 3 rows in CSV (from bottom of XML, reverse order):")
print(df[['csv_row', 'xml_position', 'sender', 'timestamp']].head(3).to_string(index=False))

print("\n🔍 Last 3 rows in CSV (going towards top of XML):")
print(df[['csv_row', 'xml_position', 'sender', 'timestamp']].tail(3).to_string(index=False))

print("\n✅ Done! CSV order: Bottom to Top (e.g., 1000 → 999 → 998 → ... → 951)")
print("   Column meanings:")
print("   - csv_row: Row number in CSV (1, 2, 3...)")
print("   - xml_position: Original position in XML file")