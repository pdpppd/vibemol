import unittest

from pymol import casp17, cmd, testing


def atom_line(serial, chain="A", resi=1, x=0.0):
    return (
        "ATOM  %5d  CA  ALA %1s%4d    %8.3f%8.3f%8.3f  1.00 12.34           C"
        % (serial, chain, resi, x, 0.0, 0.0)
    )


class FakeResponse:
    def __init__(self, status, body):
        self.status = status
        self._body = body
        self.closed = False

    def read(self):
        return self._body

    def close(self):
        self.closed = True


class FakeOpener:
    def __init__(self, response):
        self.response = response
        self.request = None

    def open(self, request, timeout=None):
        self.request = request
        return self.response


class TestCASP17(testing.PyMOLTestCase):
    def testWriteSubmission(self):
        model_lines = [
            [
                atom_line(index, "A", index, float(index)) + "EXTRA",
                "TER",
            ]
            for index in range(1, 6)
        ]

        with testing.mktemp(".pdb") as filename:
            casp17.write_submission(
                model_lines, "h1312", filename, group_id="test-group")
            text = open(filename).read()

        lines = text.splitlines()
        self.assertEqual(lines[0], "PFRMAT TS")
        self.assertEqual(lines[1], "TARGET H1312")
        self.assertEqual(lines[2], "AUTHOR test-group")
        self.assertEqual(text.count("MODEL  "), 5)
        self.assertLess(text.index("MODEL  1"), text.index("MODEL  5"))
        self.assertTrue(all(len(line) == 80 for line in lines
                            if line.startswith("ATOM")))
        self.assertEqual(lines.count("PARENT N/A"), 5)

    def testValidateKeepsFiveModels(self):
        self.assertEqual(
            casp17.validate_model_names(["m1", "m2", "m3", "m4", "m5", "m6"]),
            ["m1", "m2", "m3", "m4", "m5"])
        with self.assertRaises(casp17.CASPSubmissionError):
            casp17.validate_model_names(["m1", "m2", "m3", "m4"])

        with testing.mktemp(".pdb") as filename:
            with self.assertRaises(casp17.CASPSubmissionError):
                casp17.write_submission([[atom_line(i)] for i in range(6)],
                                        "H1312", filename,
                                        group_id="test-group")

    def testRankedModelNamesUseFirstFiveMoleculesOnly(self):
        from pymol import cgo

        for name in ["m1", "m2", "m3", "m4", "m5", "m6"]:
            cmd.fragment("gly", name)
        cmd.load_cgo([cgo.STOP], "cgo1")
        cmd.order("m3 m1 cgo1 m2 m4 m5 m6", location="top")

        self.assertEqual(
            casp17.get_ranked_model_names(cmd),
            ["m3", "m1", "m2", "m4", "m5"])

    def testBuildCurrentSubmissionUsesRankedOrder(self):
        self.assertTrue(hasattr(cmd, "casp_submit"))

        for name in ["m1", "m2", "m3", "m4", "m5", "m6"]:
            cmd.fragment("gly", name)
        cmd.order("m5 m4 m3 m2 m1 m6", location="top")
        cmd.set("casp_group_id", "test-group")

        with testing.mktemp(".pdb") as filename:
            result = casp17.build_current_submission("T1234", filename, _self=cmd)
            text = open(filename).read()

        self.assertEqual(result["model_names"], ["m5", "m4", "m3", "m2", "m1"])
        self.assertEqual(text.count("MODEL  "), 5)

    def testGroupIDSettingDrivesBuildSubmission(self):
        cmd.set("casp_group_id", "configured-group_2")

        for name in ["m1", "m2", "m3", "m4", "m5"]:
            cmd.fragment("gly", name)

        with testing.mktemp(".pdb") as filename:
            result = casp17.build_current_submission("T1234", filename, _self=cmd)
            text = open(filename).read()

        self.assertEqual(casp17.get_group_id(cmd), "configured-group_2")
        self.assertEqual(result["group_id"], "configured-group_2")
        self.assertIn("AUTHOR configured-group_2\n", text)

    def testGroupIDSettingIsRequiredForBuildSubmission(self):
        cmd.set("casp_group_id", "")

        for name in ["m1", "m2", "m3", "m4", "m5"]:
            cmd.fragment("gly", name)

        with testing.mktemp(".pdb") as filename:
            with self.assertRaises(casp17.CASPSubmissionError):
                casp17.build_current_submission("T1234", filename, _self=cmd)

    def testCustomGroupID(self):
        model_lines = [[atom_line(index)] for index in range(1, 6)]

        with testing.mktemp(".pdb") as filename:
            result = casp17.write_submission(
                model_lines, "T1234", filename, group_id="custom-group_1")
            text = open(result).read()

        self.assertIn("AUTHOR custom-group_1\n", text)

        with self.assertRaises(casp17.CASPSubmissionError):
            casp17.write_submission(
                model_lines, "T1234", filename, group_id="../bad")

        with self.assertRaises(casp17.CASPSubmissionError):
            casp17.write_submission(model_lines, "T1234", filename)

    def testSubmitToCASPUsesMultipartPost(self):
        with testing.mktemp(".pdb") as filename:
            open(filename, "w").write("PFRMAT TS\n")
            opener = FakeOpener(FakeResponse(201, b"accepted"))
            response = casp17.submit_to_casp(
                filename, "user@example.com", opener=opener)

        self.assertEqual(response, "accepted")
        self.assertIn(b'user@example.com', opener.request.data)
        self.assertIn(b'prediction_file', opener.request.data)
        self.assertIn("multipart/form-data",
                      opener.request.headers["Content-type"])

    def testSubmitToCASPRejectsNon2xx(self):
        with testing.mktemp(".pdb") as filename:
            open(filename, "w").write("PFRMAT TS\n")
            opener = FakeOpener(FakeResponse(500, b"not accepted"))
            with self.assertRaises(casp17.CASPSubmissionError):
                casp17.submit_to_casp(
                    filename, "user@example.com", opener=opener)

    def testQtGuiImportSmoke(self):
        try:
            import pymol.Qt  # noqa: F401
        except Exception as e:
            raise unittest.SkipTest(str(e))

        from pmg_qt import pymol_qt_gui

        self.assertTrue(
            hasattr(pymol_qt_gui.PyMOLQtGUI, "open_casp_submission_dialog"))
