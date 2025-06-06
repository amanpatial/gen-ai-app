"""
Data Loader Script - Pinecone Index Setup
==========================================
This script handles:
1. Creating Pinecone index
2. Loading data from folders
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
        
    def load_data_from_folder(self, folder_path="data"):
        """Load text data from files in a specified folder"""
        print(f"📂 Loading data from '{folder_path}' folder...")
        
        data = []
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"⚠️  Folder '{folder_path}' not found. Creating sample data...")
            return self.create_sample_data()
        
        # Load JSON files
        json_files = glob.glob(os.path.join(folder_path, "**/*.json"), recursive=True)
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        data.extend(file_data)
                    else:
                        data.append(file_data)
                print(f"✅ Loaded {file_path}")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        # Load TXT files
        txt_files = glob.glob(os.path.join(folder_path, "**/*.txt"), recursive=True)
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    filename = os.path.basename(file_path).replace('.txt', '')
                    data.append({
                        "id": f"txt_{filename}_{len(data)}",
                        "text": content
                    })
                print(f"✅ Loaded {file_path}")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        if not data:
            print("⚠️  No data found in folder. Creating sample data...")
            return self.create_sample_data()
        
        print(f"📊 Total loaded: {len(data)} documents")
        return data

    def create_sample_data(self):
        """Create sample data if no folder is found"""
        print("📝 Creating sample data files...")
        
        # Create data folder
        os.makedirs("data", exist_ok=True)
        
        # Sample data
        sample_data = [
            {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
            {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
            {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
            {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
            {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
            {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
        ]
        
        # Save as JSON files (split for demonstration)
        fruits_data = [item for item in sample_data if "fruit" in item["text"] or "eating" in item["text"] or "doctor" in item["text"]]
        tech_data = [item for item in sample_data if "tech" in item["text"] or "iPhone" in item["text"] or "founded" in item["text"] or "Inc." in item["text"]]
        
        # Save fruits data
        with open("data/fruits.json", 'w', encoding='utf-8') as f:
            json.dump(fruits_data, f, indent=2)
        
        # Save tech data  
        with open("data/technology.json", 'w', encoding='utf-8') as f:
            json.dump(tech_data, f, indent=2)
        
        print("✅ Sample data files created in 'data/' folder")
        return sample_data

    def create_organized_folder_structure(self):
        """Create a proper folder structure with sample files"""
        print("🗂️  Creating organized folder structure...")
        
        # Create main data folder
        os.makedirs("data", exist_ok=True)
        
        # Create subfolders
        os.makedirs("data/fruits", exist_ok=True)
        os.makedirs("data/technology", exist_ok=True)
        os.makedirs("data/general", exist_ok=True)
        
        # Sample data organized by category
        fruits_data = [
            {"id": "fruit_1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
            {"id": "fruit_2", "text": "Many people enjoy eating apples as a healthy snack."},
            {"id": "fruit_3", "text": "An apple a day keeps the doctor away, as the saying goes."},
            {"id": "fruit_4", "text": "Apples contain fiber, vitamins, and antioxidants that support good health."}
        ]
        
        tech_data = [
            {"id": "tech_1", "text": "The tech company Apple is known for its innovative products like the iPhone."},
            {"id": "tech_2", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
            {"id": "tech_3", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."},
            {"id": "tech_4", "text": "Apple's products include iPhone, iPad, Mac computers, and Apple Watch."}
        ]
        
        # Save organized data
        with open("data/fruits/apple_fruit_info.json", 'w', encoding='utf-8') as f:
            json.dump(fruits_data, f, indent=2)
        
        with open("data/technology/apple_company_info.json", 'w', encoding='utf-8') as f:
            json.dump(tech_data, f, indent=2)
        
        # Create some TXT files as examples
        with open("data/general/health_benefits.txt", 'w', encoding='utf-8') as f:
            f.write("Regular consumption of fruits like apples can contribute to overall health and wellness. They provide essential nutrients and fiber.")
        
        with open("data/general/innovation_history.txt", 'w', encoding='utf-8') as f:
            f.write("Technology companies have transformed how we communicate and work. Apple's innovations in personal computing and mobile devices have shaped modern technology.")
        
        print("✅ Organized folder structure created:")
        print("   📁 data/")
        print("   ├── 📁 fruits/")
        print("   │   └── 📄 apple_fruit_info.json")
        print("   ├── 📁 technology/") 
        print("   │   └── 📄 apple_company_info.json")
        print("   └── 📁 general/")
        print("       ├── 📄 health_benefits.txt")
        print("       └── 📄 innovation_history.txt")

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
            vectors.append({
                "id": d['id'],
                "values": e['values'],
                "metadata": {'text': d['text']}
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

    def run_data_loading_pipeline(self, create_organized_structure=False):
        """Run the complete data loading pipeline"""
        print("=" * 60)
        print("🚀 PINECONE DATA LOADER - STARTING PIPELINE")
        print("=" * 60)
        
        try:
            # Step 1: Create index
            self.create_index_if_not_exists()
            
            # Step 2: Create organized structure if requested
            if create_organized_structure:
                self.create_organized_folder_structure()
            
            # Step 3: Load data
            data = self.load_data_from_folder()
            
            if not data:
                print("❌ No data to process. Exiting...")
                return False
            
            # Step 4: Create embeddings
            embeddings = self.create_embeddings(data)
            
            # Step 5: Upsert to Pinecone
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
    
    # Ask user if they want to create organized structure
    print("Welcome to Pinecone Data Loader! 🚀")
    create_structure = input("Create organized data folder structure? (y/n): ").lower().strip()
    
    # Run the pipeline
    success = loader.run_data_loading_pipeline(
        create_organized_structure=(create_structure == 'y')
    )
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Add more documents to the 'data/' folder")
        print("2. Run this script again to update the knowledge base")
        print("3. Run 'chatbot.py' to start asking questions!")
    else:
        print("\n❌ Data loading failed. Please check your configuration.")

if __name__ == "__main__":
    main()