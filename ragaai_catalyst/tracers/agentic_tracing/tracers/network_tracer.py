from datetime import datetime
import socket
from http.client import HTTPConnection, HTTPSConnection
import aiohttp
import requests
import urllib
import uuid


class NetworkTracer:
    def __init__(self):
        self.network_calls = []
        self.patches_applied = False  # Track whether patches are active
        # Store original functions for restoration
        self._original_urlopen = None
        self._original_requests_request = None
        self._original_http_request = None
        self._original_https_request = None
        self._original_socket_create_connection = None

    def record_call(
        self,
        method,
        url,
        status_code=None,
        error=None,
        start_time=None,
        end_time=None,
        request_headers=None,
        response_headers=None,
        request_body=None,
        response_body=None,
    ):
        duration = (
            (end_time - start_time).total_seconds() if start_time and end_time else None
        )
        
        # Calculate bytes sent/received from headers and body
        bytes_sent = len(str(request_headers or "")) + len(str(request_body or ""))
        bytes_received = len(str(response_headers or "")) + len(str(response_body or ""))
        
        # Extract protocol from URL
        protocol = "https" if url.startswith("https") else "http"
            
        self.network_calls.append(
            {
                "url": url,
                "method": method,
                "status_code": status_code,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "response_time": duration,
                "bytes_sent": bytes_sent,
                "bytes_received": bytes_received,
                "protocol": protocol,
                "connection_id": str(uuid.uuid4()),  # Generate unique connection ID
                "parent_id": None,  # Will be set by the component
                "request": {
                    "headers": request_headers,
                    "body": request_body if request_body else None,
                },
                "response": {
                    "headers": response_headers,
                    "body": response_body if response_body else None,
                },
                "error": str(error) if error else None,
            }
        )

    def activate_patches(self):
        if not self.patches_applied:
            # Apply monkey patches and store originals
            self._original_urlopen = monkey_patch_urllib(self)
            self._original_requests_request = monkey_patch_requests(self)
            self._original_http_request, self._original_https_request = (
                monkey_patch_http_client(self)
            )
            self._original_socket_create_connection = monkey_patch_socket(self)
            self.patches_applied = True

    def deactivate_patches(self):
        if self.patches_applied:
            # Restore original functions
            restore_urllib(self._original_urlopen)
            restore_requests(self._original_requests_request)
            restore_http_client(
                self._original_http_request, self._original_https_request
            )
            restore_socket(self._original_socket_create_connection)
            self.network_calls = []
            self.patches_applied = False


# Define the monkey patch and restore functions
def monkey_patch_urllib(network_tracer):
    from urllib.request import urlopen

    original_urlopen = urlopen

    def patched_urlopen(url, data=None, timeout=None, *args, **kwargs):
        if isinstance(url, str):
            method = "GET" if data is None else "POST"
            url_str = url
        else:
            method = url.get_method()
            url_str = url.full_url

        start_time = datetime.now().astimezone()
        try:
            response = original_urlopen(url, data, timeout, *args, **kwargs)
            end_time = datetime.now().astimezone()
            network_tracer.record_call(
                method=method,
                url=url_str,
                status_code=response.status,
                start_time=start_time,
                end_time=end_time,
                request_headers=dict(response.request.headers),
                response_headers=dict(response.headers),
                request_body=data,
                response_body=response.read().decode("utf-8", errors="ignore"),
            )
            return response
        except Exception as e:
            end_time = datetime.now().astimezone()
            network_tracer.record_call(
                method=method,
                url=url_str,
                error=e,
                start_time=start_time,
                end_time=end_time,
            )
            raise

    urllib.request.urlopen = patched_urlopen
    return original_urlopen  # Return the original function


def restore_urllib(original_urlopen):
    urllib.request.urlopen = original_urlopen


def monkey_patch_requests(network_tracer):
    original_request = requests.Session.request

    def patched_request(self, method, url, *args, **kwargs):
        start_time = datetime.now().astimezone()
        try:
            response = original_request(self, method, url, *args, **kwargs)
            end_time = datetime.now().astimezone() 
            network_tracer.record_call(
                method=method,
                url=url,
                status_code=response.status_code,
                start_time=start_time,
                end_time=end_time,
                request_headers=dict(response.request.headers),
                response_headers=dict(response.headers),
                request_body=kwargs.get("data") or kwargs.get("json"),
                response_body=response.text,
            )
            return response
        except Exception as e:
            end_time = datetime.now().astimezone() 
            network_tracer.record_call(
                method=method,
                url=url,
                error=e,
                start_time=start_time,
                end_time=end_time,
            )
            raise

    requests.Session.request = patched_request
    return original_request


def restore_requests(original_request):
    requests.Session.request = original_request


def monkey_patch_http_client(network_tracer):
    original_http_request = HTTPConnection.request
    original_https_request = HTTPSConnection.request

    def patched_request(self, method, url, body=None, headers=None, *args, **kwargs):
        start_time = datetime.now().astimezone() 
        try:
            result = (
                original_http_request(self, method, url, body, headers, *args, **kwargs)
                if isinstance(self, HTTPConnection)
                else original_https_request(
                    self, method, url, body, headers, *args, **kwargs
                )
            )
            response = self.getresponse()
            end_time = datetime.now().astimezone() 
            network_tracer.record_call(
                method=method,
                url=f"{self._http_vsn_str} {self.host}:{self.port}{url}",
                status_code=response.status,
                start_time=start_time,
                end_time=end_time,
                request_headers=headers,
                response_headers=dict(response.headers),
                request_body=body,
                response_body=response.read().decode("utf-8", errors="ignore"),
            )
            return result
        except Exception as e:
            end_time = datetime.now().astimezone() 
            network_tracer.record_call(
                method=method,
                url=f"{self._http_vsn_str} {self.host}:{self.port}{url}",
                error=e,
                start_time=start_time,
                end_time=end_time,
            )
            raise

    HTTPConnection.request = patched_request
    HTTPSConnection.request = patched_request
    return original_http_request, original_https_request


def restore_http_client(original_http_request, original_https_request):
    HTTPConnection.request = original_http_request
    HTTPSConnection.request = original_https_request


def monkey_patch_socket(network_tracer):
    original_create_connection = socket.create_connection

    def patched_create_connection(address, *args, **kwargs):
        host, port = address
        start_time = datetime.now().astimezone() 
        try:
            result = original_create_connection(address, *args, **kwargs)
            end_time = datetime.now().astimezone()
            network_tracer.record_call(
                method="CONNECT",
                url=f"{host}:{port}",
                start_time=start_time,
                end_time=end_time,
            )
            return result
        except Exception as e:
            end_time = datetime.now().astimezone() 
            network_tracer.record_call(
                method="CONNECT",
                url=f"{host}:{port}",
                error=e,
                start_time=start_time,
                end_time=end_time,
            )
            raise

    socket.create_connection = patched_create_connection
    return original_create_connection


def restore_socket(original_create_connection):
    socket.create_connection = original_create_connection


async def patch_aiohttp_trace_config(network_tracer):
    async def on_request_start(session, trace_config_ctx, params):
        trace_config_ctx.start = datetime.now().astimezone()

    async def on_request_end(session, trace_config_ctx, params):
        end_time = datetime.now().astimezone() 
        response = params.response
        network_tracer.record_call(
            method=params.method,
            url=str(params.url),
            status_code=response.status,
            start_time=trace_config_ctx.start,
            end_time=end_time,
            request_headers=dict(params.headers),
            response_headers=dict(response.headers),
            request_body=await params.response.text(),
            response_body=await response.text(),
        )

    trace_config = aiohttp.TraceConfig()
    trace_config.on_request_start.append(on_request_start)
    trace_config.on_request_end.append(on_request_end)
    return trace_config
