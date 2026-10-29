import hashlib
import re


def get_student_params(student_id: str):
    if not re.fullmatch(r"\d{8}", student_id):
        raise ValueError("Student ID must contain exactly 8 digits")

    h = int(
        hashlib.sha256(student_id.encode("utf-8")).hexdigest()[:8],
        16,
    )

    return {
        "student_id": student_id,
        "N": 1024 + (h % 9) * 64,
        "buffer_elements": 16_777_216,
        "seed": h ^ 100007,
    }
