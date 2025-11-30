#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Weather Server - HTTP 버전
JSON-RPC 2.0 프로토콜을 사용하여 날씨 정보를 제공합니다.
"""
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Union
import uvicorn

# ─── JSON-RPC 2.0 Schemas ───────────────────────────────────────────────────
class JSONRPCRequest(BaseModel):
    jsonrpc: str
    method: str
    id: Optional[Union[str, int]] = None
    params: Optional[Dict[str, Any]] = None

class JSONRPCResponse(BaseModel):
    jsonrpc: str
    id: Optional[Union[str, int]]
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

# ─── FastAPI App ────────────────────────────────────────────────────────────
app = FastAPI(title="Weather MCP Server")

def get_weather_data(location: str) -> str:
    """더미 날씨 데이터를 반환합니다."""
    location_lower = location.lower().strip()
    
    # 더미 날씨 데이터
    weather_data = {
        "nyc": "뉴욕의 현재 날씨: 화씨 58도 (섭씨 14도), 맑음",
        "new york": "뉴욕의 현재 날씨: 화씨 58도 (섭씨 14도), 맑음",
        "london": "런던의 현재 날씨: 화씨 48도 (섭씨 9도), 흐림",
        "san francisco": "샌프란시스코의 현재 날씨: 화씨 62도 (섭씨 17도), 안개",
        "seoul": "서울의 현재 날씨: 화씨 45도 (섭씨 7도), 맑음",
    }
    
    # 부분 매칭 시도
    for key, value in weather_data.items():
        if key in location_lower or location_lower in key:
            return value
    
    return f"{location}의 날씨 정보를 찾을 수 없습니다. 대략 화씨 65도 (섭씨 18도)입니다."

def handle_jsonrpc_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-RPC 2.0 요청을 처리합니다."""
    jsonrpc = request.get("jsonrpc")
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params", {})
    
    # JSON-RPC 2.0 검증
    if jsonrpc != "2.0":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32600,
                "message": "Invalid Request: jsonrpc must be '2.0'"
            }
        }
    
    # 메서드별 처리
    try:
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "weather-server",
                        "version": "1.0.0"
                    }
                }
            }
        
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "weather",
                            "description": "특정 위치의 날씨 정보를 제공합니다. 예: NYC, London, Seoul",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "날씨를 조회할 위치"
                                    }
                                },
                                "required": ["location"]
                            }
                        }
                    ]
                }
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name == "weather":
                location = arguments.get("location", "")
                if not location:
                    # 쿼리에서 위치 추출 시도
                    query = arguments.get("query", "")
                    if "weather in" in query.lower():
                        location = query.lower().split("weather in")[1].strip().rstrip("?").strip()
                    else:
                        location = query
                
                weather_info = get_weather_data(location)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": weather_info
                            }
                        ]
                    }
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
        
        elif method == "notifications/initialized":
            # Notification은 응답하지 않음
            return None
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
    
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }

@app.post("/mcp")
async def handle_mcp(request: Request):
    """
    MCP JSON-RPC 2.0 요청을 처리합니다.
    """
    try:
        body = await request.json()
        
        # Notification인 경우 (id가 없는 경우) 응답하지 않음
        if "id" not in body or body.get("id") is None:
            return {"jsonrpc": "2.0"}
        
        response = handle_jsonrpc_request(body)
        
        if response is None:
            return {"jsonrpc": "2.0"}
        
        return response
        
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32700,
                "message": f"Parse error: {str(e)}"
            }
        }

@app.get("/")
async def root():
    """서버 상태 확인"""
    return {
        "status": "running",
        "server": "MCP Weather Server",
        "version": "1.0.0",
        "protocol": "JSON-RPC 2.0"
    }

if __name__ == "__main__":
    # Run with: python3 MCP_weather_server.py
    print("Starting MCP Weather Server on http://0.0.0.0:8000")
    print("MCP endpoint: http://0.0.0.0:8000/mcp")
    uvicorn.run(app, host="0.0.0.0", port=8000)