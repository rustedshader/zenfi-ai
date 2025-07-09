# Zenfi AI

🤖 **Advanced AI Financial Assistant for Smart Investment Decisions**

Zenfi AI is a sophisticated financial AI assistant that combines real-time market data, portfolio management, and intelligent analysis to help users make informed investment decisions. Built with modern technologies and powered by advanced AI models, Zenfi AI provides comprehensive financial insights and personalized recommendations.

**🔥 Core Capabilities:**
- **Advanced Python Analysis**: Dynamic code generation with intelligent retry mechanisms for complex financial calculations
- **Robust RAG System**: Retrieval-Augmented Generation for processing financial documents and knowledge bases
- **Complex Query Processing**: Handle sophisticated requests from simple stock prices to multi-step backtesting analysis
- **Real-time Code Execution**: Generate, execute, and refine Python code automatically until successful completion

## 🎯 Query Examples & Capabilities

Zenfi AI can handle complex financial queries and analysis tasks. Here are some examples of what you can ask:

### 📈 **Stock Analysis & Market Data**
- *"What is the stock price of Reliance?"*
- *"Analyze Apple stock performance over the last 1 month"*
- *"Get the latest Bitcoin price and create a trend analysis"*
- *"Recommend a momentum trading strategy for Apple stock around the 2025 WWDC event"*

### 📊 **Advanced Financial Analysis**
- *"Conduct a comprehensive financial analysis of Tesla for FY 2024, covering vertical/horizontal analysis, key financial ratios, cash flow and DuPont analysis"*
- *"Compare unemployment rates between different countries"*
- *"Does events happening in middle east have impact on my finance portfolio?"*

### 🔢 **Mathematical & Statistical Modeling**
- *"Calculate compound interest for $1000 at 5% for 10 years"*
- *"Generate a linear regression model with sample data"*
- *"What is the difference between ordinary least squares and maximum likelihood estimator?"*

### 📈 **Predictive Analysis with Custom Data**
```
Day,Trading Volume (thousands of shares),Closing Price (USD)
1,100,50.2
2,120,51.5
3,90,49.8
...
What will the stock price be if the trading volume is 135 thousand shares?
```

### 🗞️ **Market Intelligence**
- *"What is the latest financial news?"*
- *"What are stocks?"*
- *"Backtest the past 5 WWDC events (2019-2024) using historic stock prices from Yahoo Finance"*

## ✨ Features

### 🔍 **Real-time Stock Data & Analysis**
- **Live Stock Prices**: Get current stock prices, day high/low, and trading volumes
- **Technical Indicators**: Access 50-day and 200-day moving averages
- **Market Data**: Stock exchange info, market cap, currency, and timezone data
- **Historical Data**: Retrieve historical stock prices and performance data
- **Stock Metrics**: Track price changes, percentage changes, and year-over-year performance

### 💬 **AI-Powered Chat Assistant**
- **Natural Language Processing**: Ask questions in plain English about stocks and investments
- **Intelligent Search**: Enhanced analysis with web search capabilities for market research
- **Advanced Python Code Execution**: Run sophisticated financial calculations and analysis in real-time
- **Multi-tool Integration**: Access to comprehensive financial tools and data sources
- **Streaming Responses**: Real-time AI responses with streaming capabilities

### 🧠 **Advanced AI Analysis Engine**
- **Robust RAG (Retrieval-Augmented Generation)**: Intelligent document processing and knowledge retrieval
- **Python Code Generation & Retry Logic**: Automatically generates and refines Python code until successful execution
- **Complex Query Processing**: Handle sophisticated financial analysis requests with multi-step reasoning
- **Statistical Modeling**: Generate regression models, perform statistical analysis, and create predictive models
- **Data Processing**: Process and analyze large datasets, CSV files, and financial documents

### 📊 **Financial Tools & Data Sources**
- **Yahoo Finance Integration**: Complete stock data including prices, volumes, and financial metrics
- **Income Statement Data**: Access to company financial statements and key metrics
- **Options Chain Data**: View options contracts and related information
- **Web Search Integration**: Real-time market news and research capabilities
- **File Upload Support**: Process and analyze financial documents and attachments

### 🏦 **Portfolio Management System**
- **Asset Tracking**: Store and manage investment portfolios with detailed asset information
- **Multi-Portfolio Support**: Create and manage multiple investment portfolios
- **Asset Details**: Track asset type, quantity, purchase price, and current value
- **Portfolio Analytics**: Monitor portfolio performance and asset allocation
- **Knowledge Base**: Store and organize investment research and documents

### 🔐 **Security & Authentication**
- **JWT Authentication**: Secure user authentication with token-based system
- **Protected Routes**: Secure API endpoints with proper authorization
- **User Management**: Complete user registration and login system
- **Data Privacy**: Secure data storage with proper user isolation

## 🏗️ Architecture

### **Frontend**
- **Next.js 15**: Modern React framework with server-side rendering
- **TypeScript**: Type-safe development experience
- **Tailwind CSS**: Utility-first CSS framework for responsive design
- **React Context**: State management for authentication and data

### **Backend**
- **FastAPI**: High-performance Python web framework
- **PostgreSQL**: Robust relational database for data storage
- **SQLModel**: Python SQL toolkit with type checking
- **LangChain**: AI framework for language model integration
- **Alembic**: Database migration management

### **AI & Data**
- **Google Gemini**: Advanced language model for financial analysis
- **YFinance**: Real-time stock data and financial information
- **LangGraph**: Workflow orchestration for AI agents
- **Custom Financial Tools**: Specialized tools for stock analysis

## 🚀 Quick Start

For detailed installation and setup instructions, please refer to:
**[📖 Installation & Setup Guide](docs/README.md)**

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Next.js, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python, SQLModel |
| Database | PostgreSQL |
| AI/ML | LangChain, Google Gemini, LangGraph, Python Code Execution |
| Authentication | JWT, bcrypt |
| Data Sources | Yahoo Finance, Google Search |
| Deployment | Docker, Docker Compose |

## 📁 Project Structure

```
zenfi-ai/
├── frontend/           # Next.js React application
│   ├── src/
│   │   ├── app/       # App router and pages
│   │   ├── components/ # Reusable UI components
│   │   ├── contexts/  # React context providers
│   │   └── lib/       # Utility functions and API clients
│   └── public/        # Static assets
├── backend/           # FastAPI Python application
│   ├── app/
│   │   ├── api/       # API routes and endpoints
│   │   ├── models/    # Database models
│   │   ├── services/  # Business logic and AI services
│   │   ├── utils/     # Utility functions and tools
│   │   └── settings/  # Configuration management
│   └── migrations/    # Database migration scripts
└── docs/             # Documentation
```

## 🤝 Contributing

We welcome contributions to Zenfi AI! Please read our contributing guidelines and submit pull requests for any improvements.

## 🔗 Links

- [Installation Guide](docs/README.md)
- [API Documentation](docs/API.md)

## 📞 Support

For support and questions, please open an issue in the GitHub repository or contact our development team.

---

**⚠️ Disclaimer**: Zenfi AI provides general financial information and analysis tools. This is not personalized financial advice. Always consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results, and all investments carry risk.