import re

PATTERN = re.compile(r"\{\{([^\}]+)\}\}")


class TemplateValidationError(Exception):
    """Raised when the arguments to render a template fail to validate"""


class Template:
    """
    Jinja2-like class for string templating

    Attributes:
        content (str): original template string
        _to_render (list[str]): fields of the string that have to be rendered with the template
    """

    def __init__(self, content: str):
        """
        Create a template from a string.

        Args:
            content (str): the template string
        """
        self.content = content
        self._to_render = PATTERN.findall(content)

    def _validate(self, args: dict[str, str]) -> bool:
        return all(el in args for el in self._to_render) and all(
            isinstance(args[k], str) for k in args
        )

    def render(self, args: dict[str, str]) -> str:
        """
        Render the template.

        Args:
            args (dict[str, str]): a dictionary of arguments for the template to be rendered. The keys represent the fields in the template, and the values represent the strings with which to fill the template.

        Returns:
            str: The rendered template string.
        """
        if self._validate(args):
            content = self.content
            for word in self._to_render:
                content = content.replace("{{" + word + "}}", args[word])
            return content
        else:
            if (ls := list(set(self._to_render) - set(list(args.keys())))) != []:
                raise TemplateValidationError(
                    f"Missing the following arguments for the template: {', '.join(ls)}"
                )
            else:
                raise TemplateValidationError(
                    "You should provide a dictionary with only string values."
                )
