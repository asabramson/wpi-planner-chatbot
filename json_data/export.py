import pandas as pd
import re
import html
import json

def clean_html(raw_html):
    if not isinstance(raw_html, str):
        return str(raw_html)
    
    raw_html = raw_html.replace('\u2019', "'")
    
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    
    cleantext = html.unescape(cleantext)
    
    cleantext = re.sub(r'(Cat\. [IVX0-9]+)([A-Za-z])', r'\1 \2', cleantext)
    
    cleantext = cleantext.replace('/', '\\')
    
    cleantext = re.sub(r'\s+', ' ', cleantext).strip()
    
    return cleantext

def extract_subject(title):
    if not isinstance(title, str):
        return ""
    parts = title.split()
    if parts:
        return parts[0]
    return ""

def extract_course_code(title):
    if not isinstance(title, str):
        return ""
    parts = title.split('-')
    if parts:
        code_part = parts[0].strip()
        return code_part.replace(" ", "")
    return ""

def extract_course_title_only(title):
    if not isinstance(title, str):
        return ""
    parts = title.split('-', 1)
    if len(parts) > 1:
        return parts[1].strip()
    return title

def main():
    input_file = 'prod-data-raw.json'
    output_file = 'output.json'

    try:
        print(f"Reading {input_file}...")
        df = pd.read_json(input_file)
        
        if 'Report_Entry' in df.columns:
            print("Normalizing nested JSON data...")
            df = pd.json_normalize(df['Report_Entry'])

        section_columns = [
            'Offering_Period', 
            'Meeting_Patterns', 
            'Section_Status', 
            'Instructors', 
            'Delivery_Mode', 
            'Section_Details',
            'Meeting_Day_Patterns',
            'Instructional_Format'
        ]

        if 'Course_Title' in df.columns:
            df['Subject'] = df['Course_Title'].apply(extract_subject)
            df['Course_Code'] = df['Course_Title'].apply(extract_course_code)
            df['Clean_Title'] = df['Course_Title'].apply(extract_course_title_only)

        cols_to_clean = ['Course_Description', 'Course_Section_Description', 'Public_Notes']
        for col in cols_to_clean:
            if col in df.columns:
                df[col] = df[col].apply(clean_html)

        df.fillna('', inplace=True)

        print(f"Writing to {output_file}...")
        
        output_data = {}
        
        if 'Course_Code' in df.columns:
            for course_code, course_group in df.groupby('Course_Code'):
                first_row = course_group.iloc[0]
                
                course_obj = {
                    "level": first_row.get('Academic_Level', ''),
                    "title": first_row.get('Clean_Title', ''),
                    "discipline": first_row.get('Subject', ''),
                    "description": first_row.get('Course_Description', ''),
                    "sections": {}
                }
                
                for _, row in course_group.iterrows():
                    section_obj = {}
                    
                    term_raw = row.get('Offering_Period', '')
                    term_group = "Other"
                    if isinstance(term_raw, str):
                         match = re.search(r'(Fall|Spring|Summer)\s+([A-Z])', term_raw)
                         if match:
                             term_group = f"{match.group(1)} {match.group(2)}"
                         else:
                             term_group = term_raw
                    
                    if term_group not in course_obj['sections']:
                        course_obj['sections'][term_group] = []

                    for col in section_columns:
                        if col in df.columns:
                            key_map = {
                                'Offering_Period': 'term',
                                'Meeting_Patterns': 'time',
                                'Section_Status': 'status',
                                'Instructors': 'instructor',
                                'Delivery_Mode': 'delivery_mode',
                                'Section_Details': 'details',
                                'Meeting_Day_Patterns': 'days',
                                'Instructional_Format': 'format'
                            }
                            key = key_map.get(col, col)
                            section_obj[key] = row[col]
                    course_obj['sections'][term_group].append(section_obj)
                
                discipline = course_obj.get('discipline', 'Unknown')
                if discipline not in output_data:
                    output_data[discipline] = {}
                output_data[discipline][course_code] = course_obj
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
            
        print("Done.")

    except ValueError as e:
        print(f"Error reading JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
