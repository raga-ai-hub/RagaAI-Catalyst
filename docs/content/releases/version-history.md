# Version History

## Latest Release

### 1.2.0 (Current)
**Release Date:** October 28, 2024

#### Major Features
- Add Flask Server to server API Endpoints for Database access, removing the dependence on frontend to 
- Add Analytics Page to view Trace Analytics using new API endpoints for database access.
- Rewrote the Trace History page to use APIs for database access
- Add Trace Details Panel to `Trace History` and `Evaluation` pages 
- Improved Evaluation Metrics
- Enhanced various dashboard components
- Improved error handling
- Add more Documentation
- Add integration examples for LangGraph, OpenAI Swarm

#### Minor Improvements
- Performance optimizations
- Better error messages
- Documentation updates
- Bug fixes

## Previous Releases

### 1.1.2
**Release Date:** October 21, 2024

#### Changes
- Moved travel_planner.ipynb to examples/travel_planner.ipynb
- Various bug fixes and improvements

### 1.1.1
**Release Date:** October 17, 2024

#### Major Changes
- Added evaluation metrics support
- Enhanced UI improvements
- Added new examples (CrewAI, Autogen, Digital Marketing)
- Improved tracing capabilities

#### Minor Improvements
- Added GPU and disk info collection
- Enhanced output formatting
- Added tool_call and function_call info in trace_llm output
- Fixed tracer stop error in FinancialAnalysisSystem.ipynb
- Removed npm and nodejs dependencies
- Added wrapt dependency

### 1.1.0
**Release Date:** October 17, 2024

#### Features
- Interactive dashboard
- SQLite storage backend
- Enhanced tracing capabilities
- Tool usage analytics

#### Bug Fixes
- Fixed memory leak in long-running traces
- Corrected timestamp handling
- Resolved concurrent tracing issues

### 1.0.0
**Release Date:** September 28, 2024

#### Initial Features
- Basic tracing functionality
- LLM call monitoring
- Tool and agent tracing
- Simple metrics collection