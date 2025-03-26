import os
import xml.etree.ElementTree as ET

# Define the directory containing your .world files
directory = '../worlds/'

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.world'):
        file_path = os.path.join(directory, filename)
        print(f'Processing file: {file_path}')

        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Namespace handling (if any)
        namespaces = {'sdf': 'http://sdformat.org/schemas/root.xsd'}

        # Find all <cast_shadows> elements
        for cast_shadows in root.findall('.//cast_shadows', namespaces):
            if cast_shadows.text == '1':
                print(f' - Changing <cast_shadows>1</cast_shadows> to <cast_shadows>0</cast_shadows>')
                cast_shadows.text = '0'

        # Write the modified XML back to the file
        tree.write(file_path, encoding='UTF-8', xml_declaration=True)

        print(f' - File updated successfully.\n')

