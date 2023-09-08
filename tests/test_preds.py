"""Test cases for utilities related to our clorm fact format."""
from unittest import TestCase

from clingo.symbol import String
from clorm import StringField, refine_field

import renopro.predicates as preds


class TestPredUtils(TestCase):
    """Test additional utilities for working with clorm predicates."""

    def test_combine_fieleds_lazily_non_field_error(self):
        """Combining a non BaseField (sub)class should raise error"""
        message = "{preds.Function1} is not a BaseField or a sub-class."
        with self.assertRaises(TypeError, msg=message):
            preds.combine_fields_lazily(
                [preds.id_terms.Variable.Field, preds.id_terms.Function]
            )

    def test_combine_fields_lazily_no_combined_pytocl_error(self):
        """If python data cannot be converted to clingo data by any of
        the combined fields, should raise a TypeError.

        """
        asd_field = refine_field(StringField, ["asd"])
        dsa_field = refine_field(StringField, ["dsa"])
        AsdOrDsaField = preds.combine_fields_lazily(
            [asd_field, dsa_field], name="AsdOrDsaField"
        )

        py_str = "banana"
        message = f"No combined pytocl() match for value {py_str}."
        with self.assertRaises(TypeError, msg=message):
            AsdOrDsaField.pytocl(py_str)

    def test_combine_fields_lazily_failure_to_unify(self):
        """If clingo data cannot be converted to python data by any of
        the combined fields, should raise a TypeError.

        """
        asd_field = refine_field(StringField, ["asd"])
        dsa_field = refine_field(StringField, ["dsa"])
        asd_dsa_field = preds.combine_fields_lazily([asd_field, dsa_field])
        clingo_str = String("banana")
        message = (
            f"Object '{clingo_str}' ({type(clingo_str)}) failed to "
            f"unify with AsdOrDsaField."
        )
        with self.assertRaises(TypeError, msg=message):
            asd_dsa_field.cltopy(clingo_str)
