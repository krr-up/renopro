"""Test cases for utilities related to our clorm fact format."""
from unittest import TestCase

from clorm import StringField, refine_field
from clingo.symbol import String

import renopro.predicates as preds


class TestCombineFieldsLazily(TestCase):

    def test_non_BaseField_error(self):
        """ Combining a non BaseField (sub)class should raise error"""
        message = "{preds.Function1} is not a BaseField or a sub-class."
        with self.assertRaises(TypeError, msg=message):
            preds.combine_fields_lazily(
                [preds.Variable1.Field, preds.Function1])

    def test_no_combined_pytocl_error(self):
        """If python data cannot be converted to clingo data by any of
        the combined fields, should raise a TypeError.

        """
        AsdField = refine_field(StringField, ["asd"])
        DsaField = refine_field(StringField, ["dsa"])
        AsdOrDsaField = preds.combine_fields_lazily(
            [AsdField, DsaField], name="AsdOrDsaField")

        py_str = "banana"
        message = f"No combined pytocl() match for value {py_str}."
        with self.assertRaises(TypeError, msg=message):
            AsdOrDsaField.pytocl(py_str)

    def test_failure_to_unify(self):
        """If clingo data cannot be converted to python data by any of
        the combined fields, should raise a TypeError.

        """
        AsdField = refine_field(StringField, ["asd"])
        DsaField = refine_field(StringField, ["dsa"])
        AsdOrDsaField = preds.combine_fields_lazily(
            [AsdField, DsaField])
        clingo_str = String("banana")
        message = (f"Object '{clingo_str}' ({type(clingo_str)}) failed to "
                   f"unify with AsdOrDsaField.")
        with self.assertRaises(TypeError, msg=message):
            AsdOrDsaField.cltopy(clingo_str)
