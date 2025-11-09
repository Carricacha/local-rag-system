@'
"""
Obsidian Document Loader
Loads markdown files from Obsidian vault via REST API
"""

import requests
from langchain.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()

class ObsidianDocumentLoader:
    """Load documents from Obsidian vault via REST API"""
    
    def __init__(self):
        self.api_url = os.getenv("OBSIDIAN_API_URL")
        self.api_key = os.getenv("OBSIDIAN_API_KEY")
        self.vault_name = os.getenv("OBSIDIAN_VAULT_NAME")
    
    def load_from_path(self, folder_path: str) -> list:
        """Load markdown files from specific folder"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = requests.get(
                f"{self.api_url}/vault",
                headers=headers
            )
            
            if response.status_code != 200:
                print(f"❌ Error connecting to Obsidian: {response.status_code}")
                return []
            
            documents = []
            files = response.json()
            
            for file in files:
                if file["path"].startswith(folder_path) and file["path"].endswith(".md"):
                    try:
                        file_response = requests.get(
                            f"{self.api_url}/vault/read?file={file['path']}",
                            headers=headers
                        )
                        
                        if file_response.status_code == 200:
                            content = file_response.json()["content"]
                            
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source": file["path"],
                                    "vault": self.vault_name,
                                    "type": "documentation"
                                }
                            )
                            documents.append(doc)
                            print(f"✅ Loaded: {file['path']}")
                    
                    except Exception as e:
                        print(f"⚠️ Error loading {file['path']}: {e}")
            
            return documents
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def load_all_vault(self) -> list:
        """Load all markdown files from entire vault"""
        return self.load_from_path("")
'@ | Out-File -FilePath obsidian_loader.py -Encoding UTF8
