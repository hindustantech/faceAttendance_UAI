# app/middleware/auth.py
from fastapi import Header, HTTPException, Request
from app.config import settings
from app.utils.logger import setup_logger
from functools import wraps

logger = setup_logger(__name__)

async def verify_api_key(
    request: Request,
    x_api_key: str = Header(None, alias="X-API-Key")
):
    """
    Verify API key from request header
    
    This middleware ensures only authorized services can access the API
    """
    if not x_api_key:
        logger.warning(f"Missing API key - IP: {request.client.host}")
        raise HTTPException(
            status_code=401,
            detail={
                "message": "API key is required",
                "error_code": "MISSING_API_KEY"
            }
        )
    
    if x_api_key != settings.API_KEY:
        logger.warning(f"Invalid API key attempt - IP: {request.client.host}")
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Invalid API key",
                "error_code": "INVALID_API_KEY"
            }
        )
    
    return x_api_key