import base64
import hashlib
import hmac

from src.auth_token_generator import AuthTokenGenerator


def test_calc_xauth_token_format() -> None:
    token = AuthTokenGenerator().calc_xauth_token("testuser", "secretkey")
    encoded_key, hmac_part = token.split(".")

    assert base64.b64decode(encoded_key).decode() == "secretkey"

    expected_hmac = base64.b64encode(
        hmac.new(b"secretkey", msg=b"testuser", digestmod=hashlib.sha256).digest()
    ).decode()
    assert hmac_part == expected_hmac
