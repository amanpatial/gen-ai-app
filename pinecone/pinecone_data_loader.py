"""
Data Loader Script - Pinecone Index Setup (PDF, Text, JSON Only)
================================================================
This script handles:
1. Creating Pinecone index
2. Loading data from PDF, TXT, and JSON files only
3. Creating embeddings
4. Upserting data to Pinecone

Run this script first to set up your knowledge base.
"""

import os
import time
import json
import glob
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# PDF processing imports
try:
    import PyPDF2
except ImportError:
    print("⚠️  PyPDF2 not found. Install with: pip install PyPDF2")
    PyPDF2 = None

try:
    from pypdf import PdfReader
except ImportError:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("⚠️  No PDF library found. Install PyPDF2, pypdf, or PyMuPDF")
        fitz = None

# Load environment variables
load_dotenv()

class PineconeDataLoader:
    def __init__(self):
        """Initialize Pinecone client and configuration"""
        self.pine_cone_api_key = os.environ["PINE_CONE_API_KEY"]
        self.pc = Pinecone(api_key=self.pine_cone_api_key)
        self.index_name = "aman-hello-advance-index-1"
        self.namespace = "ns1"
        
    def create_index_if_not_exists(self):
        """Create Pinecone index if it doesn't exist"""
        print("🔍 Checking if index exists...")
        
        if self.index_name not in [index.name for index in self.pc.list_indexes()]:
            print(f"📝 Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"✅ Created index: {self.index_name}")
        else:
            print(f"✅ Index '{self.index_name}' already exists")
        
        # Wait for index to be ready
        print("⏳ Waiting for index to be ready...")
        while not self.pc.describe_index(self.index_name).status['ready']:
            time.sleep(1)
        
        self.index = self.pc.Index(self.index_name)
        print("🚀 Index is ready!")

    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file using available PDF library"""
        try:
            # Try PyPDF2 first
            if PyPDF2:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text.strip()
            
            # Try pypdf
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
            except ImportError:
                pass
            
            # Try PyMuPDF
            if fitz:
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text() + "\n"
                doc.close()
                return text.strip()
            
            print(f"❌ No PDF library available to process {file_path}")
            return None
            
        except Exception as e:
            print(f"❌ Error extracting text from PDF {file_path}: {e}")
            return None
        
    def load_data_from_folder(self, folder_path="data"):
        """Load data from PDF, TXT, and JSON files only"""
        print(f"📂 Loading data from '{folder_path}' folder...")
        print("📋 Supported file types: PDF, TXT, JSON")
        
        data = []
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"⚠️  Folder '{folder_path}' not found. Creating sample data...")
            return self.create_sample_data()
        
        # Load PDF files
        pdf_files = glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True)
        for file_path in pdf_files:
            try:
                content = self.extract_text_from_pdf(file_path)
                if content and content.strip():
                    filename = os.path.basename(file_path).replace('.pdf', '')
                    data.append({
                        "id": f"pdf_{filename}_{len(data)}",
                        "text": content[:2000],  # Limit to 2000 chars for embedding
                        "source": file_path,
                        "type": "pdf"
                    })
                    print(f"✅ Loaded PDF: {file_path} ({len(content)} chars)")
                else:
                    print(f"⚠️  Empty content in PDF: {file_path}")
            except Exception as e:
                print(f"❌ Error loading PDF {file_path}: {e}")
        
        # Load TXT files
        txt_files = glob.glob(os.path.join(folder_path, "**/*.txt"), recursive=True)
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        filename = os.path.basename(file_path).replace('.txt', '')
                        data.append({
                            "id": f"txt_{filename}_{len(data)}",
                            "text": content,
                            "source": file_path,
                            "type": "txt"
                        })
                        print(f"✅ Loaded TXT: {file_path}")
                    else:
                        print(f"⚠️  Empty TXT file: {file_path}")
            except Exception as e:
                print(f"❌ Error loading TXT {file_path}: {e}")
        
        # Load JSON files (limit to 1 as requested)
        json_files = glob.glob(os.path.join(folder_path, "**/*.json"), recursive=True)
        json_count = 0
        for file_path in json_files:
            if json_count >= 1:  # Limit to 1 JSON file
                print(f"⏭️  Skipping additional JSON file: {file_path}")
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        for item in file_data:
                            if isinstance(item, dict) and 'text' in item:
                                item['source'] = file_path
                                item['type'] = 'json'
                                data.append(item)
                    else:
                        if isinstance(file_data, dict) and 'text' in file_data:
                            file_data['source'] = file_path
                            file_data['type'] = 'json'
                            data.append(file_data)
                
                json_count += 1
                print(f"✅ Loaded JSON: {file_path}")
            except Exception as e:
                print(f"❌ Error loading JSON {file_path}: {e}")
        
        if not data:
            print("⚠️  No supported files found. Creating sample data...")
            return self.create_sample_data()
        
        print(f"📊 Total loaded: {len(data)} documents")
        print(f"   📄 PDF files: {sum(1 for d in data if d.get('type') == 'pdf')}")
        print(f"   📝 TXT files: {sum(1 for d in data if d.get('type') == 'txt')}")
        print(f"   📋 JSON files: {sum(1 for d in data if d.get('type') == 'json')}")
        
        return data

    def create_sample_data(self):
        """Create sample PDF, TXT, and JSON files"""
        print("📝 Creating sample data files...")
        
        # Create data folder
        os.makedirs("data", exist_ok=True)
        
        # Create sample TXT files
        txt_samples = [
            {
                "filename": "artificial_intelligence.txt",
                "content": """Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks typically requiring human intelligence. This includes learning, reasoning, problem-solving, perception, and language understanding.

Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. Deep Learning, a subset of machine learning, uses neural networks with multiple layers to model and understand complex patterns in data.

AI applications include natural language processing, computer vision, robotics, autonomous vehicles, recommendation systems, and medical diagnosis. The field continues to evolve rapidly with advancements in computing power and algorithmic improvements."""
            },
            {
                "filename": "renewable_energy.txt", 
                "content": """Renewable energy sources are naturally replenishing and environmentally sustainable alternatives to fossil fuels. The main types include solar energy, wind energy, hydroelectric power, geothermal energy, and biomass.

Solar panels convert sunlight directly into electricity using photovoltaic cells. Wind turbines harness kinetic energy from moving air to generate power. Hydroelectric dams use flowing water to produce electricity.

The transition to renewable energy is crucial for combating climate change, reducing greenhouse gas emissions, and achieving energy independence. Many countries are investing heavily in renewable infrastructure and setting ambitious clean energy targets for the coming decades."""
            },
            {
                "filename": "space_exploration.txt",
                "content": """Space exploration has been one of humanity's greatest achievements, expanding our understanding of the universe and our place within it. From the first satellite launches in the 1950s to recent Mars rover missions, we've made remarkable progress.

The International Space Station serves as a platform for scientific research in microgravity. Private companies like SpaceX have revolutionized space travel with reusable rockets, making launches more cost-effective.

Future missions include returning humans to the Moon, establishing permanent lunar bases, and eventual crewed missions to Mars. Space exploration drives technological innovation that benefits life on Earth in areas such as communications, materials science, and medical technology."""
            }
        ]
        
        # Create TXT files
        for sample in txt_samples:
            with open(f"data/{sample['filename']}", 'w', encoding='utf-8') as f:
                f.write(sample['content'])
            print(f"✅ Created: data/{sample['filename']}")
        
        # Create sample JSON file
        json_data = [
            {
                "id": "tech_1",
                "text": "Cloud computing delivers computing services including servers, storage, databases, networking, software, analytics, and intelligence over the Internet to offer faster innovation, flexible resources, and economies of scale."
            },
            {
                "id": "tech_2", 
                "text": "Blockchain technology is a decentralized, distributed ledger that records transactions across multiple computers in a way that makes it difficult to change, hack, or cheat the system."
            },
            {
                "id": "tech_3",
                "text": "Internet of Things (IoT) refers to the network of physical objects embedded with sensors, software, and other technologies for connecting and exchanging data with other devices and systems over the internet."
            },
            {
                "id": "tech_4",
                "text": "Cybersecurity involves protecting systems, networks, and programs from digital attacks. These attacks usually aim to access, change, or destroy sensitive information, extort money, or interrupt normal business processes."
            }
        ]
        
        with open("data/technology_concepts.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        print("✅ Created: data/technology_concepts.json")
        
        # Create sample PDF content instruction
        print("\n📋 Sample PDF Creation Instructions:")
        print("=" * 50)
        print("To complete the sample data setup, please:")
        print("1. Create 2-3 PDF files and place them in the 'data/' folder")
        print("2. Suggested PDF topics:")
        print("   • quantum_computing.pdf")
        print("   • climate_change.pdf") 
        print("   • digital_transformation.pdf")
        print("3. You can create PDFs from any text editor or online PDF generator")
        print("4. Alternatively, download sample PDFs from online sources")
        print("\n💡 The script will automatically process any PDF files found in the data folder.")
        
        # Return the loaded data for immediate processing
        return self.load_data_from_folder()

    def create_embeddings(self, data):
        """Create embeddings for the loaded data"""
        print("🧠 Creating embeddings...")
        
        embeddings = self.pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[d['text'] for d in data],
            parameters={"input_type": "passage", "truncate": "END"}
        )
        
        print(f"✅ Created {len(embeddings)} embeddings")
        print(f"📐 Embedding dimension: {len(embeddings[0]['values'])}")
        return embeddings

    def upsert_data(self, data, embeddings):
        """Upsert data to Pinecone index"""
        print("📤 Upserting data to Pinecone...")
        
        # Prepare vectors for upsert
        vectors = []
        for d, e in zip(data, embeddings):
            metadata = {
                'text': d['text'][:1000],  # Limit metadata text length
                'source': d.get('source', 'unknown'),
                'type': d.get('type', 'unknown')
            }
            
            vectors.append({
                "id": d['id'],
                "values": e['values'],
                "metadata": metadata
            })

        # Upsert data
        self.index.upsert(
            vectors=vectors,
            namespace=self.namespace
        )
        
        print("✅ Data upserted successfully!")
        
        # Show index stats
        stats = self.index.describe_index_stats()
        print(f"📊 Index stats: {stats}")

    def run_data_loading_pipeline(self):
        """Run the complete data loading pipeline"""
        print("=" * 60)
        print("🚀 PINECONE DATA LOADER - PDF, TXT, JSON ONLY")
        print("=" * 60)
        
        try:
            # Step 1: Create index
            self.create_index_if_not_exists()
            
            # Step 2: Load data (PDF, TXT, JSON only)
            data = self.load_data_from_folder()
            
            if not data:
                print("❌ No data to process. Exiting...")
                return False
            
            # Step 3: Create embeddings
            embeddings = self.create_embeddings(data)
            
            # Step 4: Upsert to Pinecone
            self.upsert_data(data, embeddings)
            
            print("=" * 60)
            print("🎉 DATA LOADING PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("✅ Your knowledge base is ready for querying!")
            print("🔥 You can now run the chatbot script to start asking questions!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in data loading pipeline: {e}")
            return False

def main():
    """Main function to run the data loader"""
    loader = PineconeDataLoader()
    
    print("Welcome to Pinecone Data Loader! 🚀")
    print("📋 This script loads PDF, TXT, and JSON files only")
    print("🔢 Limits: 2-3 PDFs, 2-3 TXT files, 1 JSON file")
    
    # Run the pipeline
    success = loader.run_data_loading_pipeline()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Add more PDF/TXT/JSON files to the 'data/' folder")
        print("2. Run this script again to update the knowledge base")
        print("3. Run 'chatbot.py' to start asking questions!")
        print("\n📚 File Type Support:")
        print("   ✅ PDF files (.pdf)")
        print("   ✅ Text files (.txt)")
        print("   ✅ JSON files (.json) - Limited to 1 file")
        print("   ❌ Other file types are ignored")
    else:
        print("\n❌ Data loading failed. Please check your configuration.")
        print("💡 Make sure you have PDF processing libraries installed:")
        print("   pip install PyPDF2 pypdf PyMuPDF")

if __name__ == "__main__":
    main()