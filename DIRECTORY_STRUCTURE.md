# 📂 Directory Structure

This document provides a visual overview of the repository's organization.

## 🗂️ Complete Directory Tree

```
AI-Development-Knowledge-Base/
│
├── 📂 .codemie/
│   └── 📂 virtual_assistants/
│       ├── api_developer.yaml
│       ├── code_debugger.yaml
│       ├── code_documenter.yaml
│       ├── code_reviewer.yaml
│       ├── code_security_auditor.yaml
│       ├── codemie_coder.yaml
│       ├── frontend_developer.yaml
│       └── python_developer.yaml
│
├── 📂 docs/
│   ├── 📂 interview-prep/
│   │   ├── AI_Python_Developer_Interview_Study_Plan.md
│   │   └── Interview_Preparation_RAG.md
│   │
│   ├── 📂 langchain/
│   │   ├── LangChain_Introduction.md
│   │   ├── LangChain_Models_and_Prompts.md
│   │   ├── LangChain_Chains.md
│   │   ├── LangChain_Agents.md
│   │   ├── LangChain_Chains_vs_Agents.md
│   │   ├── LangChain_Memory_and_Storage.md
│   │   ├── LangChain_MCP_Server.md
│   │   ├── LangChain_Advanced_Features.md
│   │   ├── LangChain_Best_Practices.md
│   │   └── LangChain_Key_Concepts.md
│   │
│   ├── 📂 rag/
│   │   ├── RAG_Fundamentals.md
│   │   ├── RAG_Multi_Agent_System.md
│   │   ├── RAG_Implementation.md
│   │   └── RAG_Best_Practices.md
│   │
│   └── 📂 resources/
│       ├── EPAM AI Run Framework.pdf
│       └── EPAM AI Run Framework.pptx
│
├── 📂 logs/
│   └── output.txt
│
├── 📄 README.md
├── 📄 README_LangChain_Guide.md
├── 📄 SEPARATION_SUMMARY.md
└── 📄 DIRECTORY_STRUCTURE.md (this file)
```

## 📊 Directory Breakdown

### `.codemie/` - AI Assistant Configurations
**Purpose:** Store CodeMie virtual assistant configurations for different development roles.

**Contents:**
- 8 YAML configuration files
- Role-specific AI assistant setups
- Pre-configured prompts and behaviors

**Use Case:** Quick setup of AI coding assistants for various development tasks.

---

### `docs/` - Documentation Hub
**Purpose:** Central location for all documentation and knowledge resources.

#### `docs/interview-prep/` (2 files)
- Interview preparation materials
- Study plans for AI/Python developers
- RAG system interview questions

#### `docs/langchain/` (10 files)
- Comprehensive LangChain framework documentation
- From beginner to advanced concepts
- Practical examples and best practices

#### `docs/rag/` (4 files)
- RAG (Retrieval-Augmented Generation) documentation
- Architecture patterns and implementations
- Multi-agent systems and best practices

#### `docs/resources/` (2 files)
- PDF and PowerPoint presentations
- EPAM AI Run Framework materials
- Reference documentation

---

### `logs/` - Application Logs
**Purpose:** Store application outputs and log files.

**Contents:**
- `output.txt` - General application output

---

## 🎯 Quick Navigation

### By Role

| Role | Primary Directory | Quick Start |
|------|------------------|-------------|
| **AI/ML Engineer** | `docs/langchain/`, `docs/rag/` | [LangChain Introduction](docs/langchain/LangChain_Introduction.md) |
| **Python Developer** | `docs/langchain/`, `.codemie/` | [Python Developer Assistant](.codemie/virtual_assistants/python_developer.yaml) |
| **Interview Candidate** | `docs/interview-prep/` | [Interview Prep RAG](docs/interview-prep/Interview_Preparation_RAG.md) |
| **API Developer** | `.codemie/`, `docs/langchain/` | [API Developer Assistant](.codemie/virtual_assistants/api_developer.yaml) |
| **QA Engineer** | `.codemie/`, `docs/langchain/` | [Code Reviewer Assistant](.codemie/virtual_assistants/code_reviewer.yaml) |

### By Topic

| Topic | Location | File Count |
|-------|----------|------------|
| **LangChain** | `docs/langchain/` | 10 files |
| **RAG Systems** | `docs/rag/` | 4 files |
| **Interview Prep** | `docs/interview-prep/` | 2 files |
| **AI Assistants** | `.codemie/virtual_assistants/` | 8 files |
| **Presentations** | `docs/resources/` | 2 files |

### By Complexity

| Level | Recommended Files |
|-------|------------------|
| **Beginner** | `docs/langchain/LangChain_Introduction.md`<br>`docs/rag/RAG_Fundamentals.md` |
| **Intermediate** | `docs/langchain/LangChain_Agents.md`<br>`docs/rag/RAG_Implementation.md` |
| **Advanced** | `docs/langchain/LangChain_Best_Practices.md`<br>`docs/rag/RAG_Multi_Agent_System.md` |

## 📈 File Statistics

```
Total Directories: 6
Total Files: 30
├── Documentation Files: 18 (.md)
├── Configuration Files: 8 (.yaml)
├── Presentation Files: 2 (.pdf, .pptx)
└── Log Files: 2 (.txt, .md)

Total Lines of Documentation: ~5,000+
Average File Size: ~300-800 lines
Largest File: LangChain_Best_Practices.md (~1,200 lines)
```

## 🔄 Recent Changes

### Reorganization (December 2024)
✅ Created `docs/resources/` directory  
✅ Moved PDF and PPTX files from root  
✅ Created `logs/` directory  
✅ Moved `output.txt` to logs/  
✅ Created comprehensive README.md  
✅ Created DIRECTORY_STRUCTURE.md  

### Benefits:
- ✨ Cleaner root directory
- 📁 Better file categorization
- 🔍 Easier navigation
- 📚 Improved discoverability
- 🎯 Clear separation of concerns

## 🛠️ Maintenance Guidelines

### Adding New Files

1. **Documentation Files (.md)**
   - LangChain related → `docs/langchain/`
   - RAG related → `docs/rag/`
   - Interview prep → `docs/interview-prep/`

2. **Resource Files (.pdf, .pptx, etc.)**
   - All resources → `docs/resources/`

3. **Log Files (.txt, .log)**
   - All logs → `logs/`

4. **Configuration Files (.yaml, .json)**
   - AI assistants → `.codemie/virtual_assistants/`

### Naming Conventions

- Use PascalCase for documentation: `RAG_Implementation.md`
- Use snake_case for configs: `python_developer.yaml`
- Use descriptive names: `LangChain_Best_Practices.md` not `best.md`
- Prefix related files: `RAG_*.md`, `LangChain_*.md`

### Documentation Updates

When adding or modifying files:
1. Update this DIRECTORY_STRUCTURE.md
2. Update main README.md
3. Update README_LangChain_Guide.md (if LangChain related)
4. Add cross-references in related documents

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | Dec 2024 | Complete reorganization, new directory structure |
| 1.0 | Nov 2024 | Initial documentation split from large files |

---

**Last Updated:** December 2024  
**Total Files:** 30  
**Total Size:** ~5,000+ lines of documentation  
**Organization Status:** ✅ Fully Organized
