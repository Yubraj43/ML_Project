from typing import Any


def error_message_details(error: Exception, error_detail: Any) -> str:
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "unknown_file"
    line_number = exc_tb.tb_lineno if exc_tb else -1
    return (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{line_number}] error message [{str(error)}]"
    )


class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: Any) -> None:
        super().__init__(str(error_message))
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
