import os

def simple_page_chunker(input_path, output_dir, pages_per_chunk=1000):
    os.makedirs(output_dir, exist_ok=True)

    chunk_id = 1
    page_count = 0
    output_file = open(os.path.join(output_dir, f'chunk_{chunk_id}.xml'), 'w', encoding='utf-8')
    output_file.write('<?xml version="1.0"?>\n<mediawiki>\n')  # Start the chunk with the root tag

    with open(input_path, 'r', encoding='utf-8') as infile:
        inside_page = False  # Flag to detect if we're inside a <page> tag
        for line in infile:
            output_file.write(line)

            # If a <page> tag is found, increment the page count
            if "<page>" in line:
                inside_page = True

            if inside_page and "</page>" in line:
                page_count += 1
                inside_page = False  # Reset after the end of the page

            # If we've reached the chunk size (i.e., 1000 pages), close this chunk and start a new one
            if page_count >= pages_per_chunk:
                output_file.write('</mediawiki>')  # Close the current chunk
                output_file.close()

                chunk_id += 1  # Increment the chunk number
                page_count = 0  # Reset the page count
                output_file = open(os.path.join(output_dir, f'chunk_{chunk_id}.xml'), 'w', encoding='utf-8')
                output_file.write('<?xml version="1.0"?>\n<mediawiki>\n')  # Open a new chunk

    # Final chunk write (close the last chunk properly)
    output_file.write('</mediawiki>')  # Close the last chunk
    output_file.close()

    print(f"✅ Done. Total chunks: {chunk_id}")

# Example Usage:
simple_page_chunker(input_path="C:/Users/gowth/Documents/rag_datasets/wikidump/enwiki-latest-pages-articles.xml",
                     output_dir='C:/Users/gowth/Documents/rag_datasets/wikidump/chunk_Folder', 
                     pages_per_chunk=2000)
