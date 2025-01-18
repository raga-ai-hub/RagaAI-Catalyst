import os
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def get_token():
#     access_key = os.getenv("RAGA-TRACER_ACCESS_KEY")
#     secret_key = os.getenv("RAGA-TRACER_SECRET_KEY")
#     headers = {"Content-Type": "application/json"}
#     json_data = {
#         "accessKey": access_key,
#         "secretKey": secret_key,
#     }
#     response = requests.post(
#         "https://backend.dev3.ragaai.ai/api/token", headers=headers, json=json_data
#     )
#     token_response = response.json()
#     token = token_response.get("data", {}).get("token", None)
#     if token is not None:
#         os.environ["RAGAAI_CATALYST_TOKEN"] = token
#     return token_response


def response_checker(response, context=""):
    """
    Checks the response status code and logs the appropriate message.

    Args:
        response (requests.Response): The response object.
        context (str, optional): The context in which the response is being checked. Defaults to "".

    Returns:
        int: The status code of the response.
    """
    logger.debug(f" Response : {response}")
    if response.status_code == 200:
        logger.debug(
            f"{context} - Successful Request. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 400:
        logger.debug(
            f"{context} - Bad Request. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 401:
        logger.debug(
            f"{context} - Unauthorized. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 403:
        logger.debug(
            f"{context} - Forbidden. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 404:
        logger.debug(
            f"{context} - Not Found. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 408:
        logger.debug(
            f"{context} - Request Timeout. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 500:
        logger.debug(
            f"{context} - Internal Server Error. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 502:
        logger.debug(
            f"{context} - Bad Gateway. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 503:
        logger.debug(
            f"{context} - Service Unavailable. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 504:
        logger.debug(
            f"{context} - Gateway Timeout. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    else:
        error_message = response.json().get("message", "No message returned.")
        logger.debug(
            f"{context}{response.reason}. Response Code: {response.status_code}, Response Text: {error_message}"
        )
        return response.status_code
