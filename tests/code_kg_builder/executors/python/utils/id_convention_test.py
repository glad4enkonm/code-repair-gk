"""Tests for ID convention utilities."""

from code_kg_builder.executors.python.utils.id_convention import (
    make_class_id,
    make_dependency_id,
    make_directory_id,
    make_decorator_id,
    make_edge_id,
    make_exception_id,
    make_file_id,
    make_function_id,
    make_import_id,
    make_method_id,
    make_type_annotation_id,
    make_variable_id,
    relative_path_str,
)


class TestRelativePath:
    def test_simple(self, tmp_path):
        f = tmp_path / "src" / "main.py"
        assert relative_path_str(tmp_path, f) == "src/main.py"

    def test_root_file(self, tmp_path):
        f = tmp_path / "setup.py"
        assert relative_path_str(tmp_path, f) == "setup.py"

    def test_nested(self, tmp_path):
        f = tmp_path / "django" / "db" / "models" / "query.py"
        assert relative_path_str(tmp_path, f) == "django/db/models/query.py"


class TestDirectoryId:
    def test_root(self):
        assert make_directory_id("") == "Directory-"

    def test_nested(self):
        assert make_directory_id("django/db/models") == "Directory-django/db/models"


class TestFileId:
    def test_simple(self):
        assert make_file_id("src/main.py") == "File-src/main.py"


class TestClassId:
    def test_simple(self):
        assert (
            make_class_id("src/views.py", "BaseView") == "Class-src/views.py-BaseView"
        )


class TestFunctionId:
    def test_simple(self):
        assert (
            make_function_id("src/utils.py", "parse") == "Function-src/utils.py-parse"
        )


class TestMethodId:
    def test_simple(self):
        assert make_method_id("src/views.py", "BaseView", "get") == (
            "Method-src/views.py-BaseView-get"
        )


class TestVariableId:
    def test_with_parent(self):
        parent = "Method-src/views.py-BaseView-get"
        assert make_variable_id(parent, "args") == (
            "Variable-Method-src/views.py-BaseView-get-args"
        )


class TestDecoratorId:
    def test_simple(self):
        target = "Method-src/views.py-BaseView-get"
        assert make_decorator_id(target, "cached_property") == (
            "Decorator-Method-src/views.py-BaseView-get-cached_property"
        )


class TestImportId:
    def test_simple(self):
        fid = make_file_id("src/views.py")
        assert make_import_id(fid, 0) == "Import-File-src/views.py-0"

    def test_indexed(self):
        fid = make_file_id("src/views.py")
        assert make_import_id(fid, 3) == "Import-File-src/views.py-3"


class TestDependencyId:
    def test_simple(self):
        assert make_dependency_id("django") == "Dependency-django"


class TestExceptionId:
    def test_simple(self):
        assert make_exception_id("FieldError") == "Exception-FieldError"


class TestTypeAnnotationId:
    def test_simple(self):
        assert make_type_annotation_id("QuerySet") == "TypeAnnotation-QuerySet"


class TestEdgeId:
    def test_simple(self):
        eid = make_edge_id(
            "Class-src/views.py-BaseView",
            "CONTAINS",
            "Method-src/views.py-BaseView-get",
        )
        assert eid == (
            "Class-src/views.py-BaseView-CONTAINS-Method-src/views.py-BaseView-get"
        )
