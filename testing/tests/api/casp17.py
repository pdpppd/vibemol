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
    def load_ranked_models(self, names=("m1", "m2", "m3", "m4", "m5")):
        casp17.reset_ranked_model_names(cmd)
        for index, name in enumerate(names, start=1):
            cmd.fragment("gly", name)
            casp17.assign_ranked_model(index, name, cmd)
        return list(names)

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

    def testRankedModelNamesStartBlank(self):
        casp17.reset_ranked_model_names(cmd)
        self.assertEqual(casp17.get_ranked_model_names(cmd), [None] * 5)
        with self.assertRaises(casp17.CASPSubmissionError):
            casp17.get_validated_ranked_model_names(cmd)

    def testCaspModeEnableResetsRankList(self):
        cmd.fragment("gly", "m1")
        casp17.reset_ranked_model_names(cmd)
        casp17.assign_ranked_model(1, "m1", cmd)
        cmd.set("casp_submission_mode", 0)
        self.assertEqual(casp17.get_ranked_model_names(cmd)[0], "m1")

        try:
            cmd.set("casp_submission_mode", 1)
            self.assertEqual(casp17.get_ranked_model_names(cmd), [None] * 5)
        finally:
            cmd.set("casp_submission_mode", 0)

    def testAssignRankedModels(self):
        for name in ["m1", "m2", "m3"]:
            cmd.fragment("gly", name)

        casp17.reset_ranked_model_names(cmd)
        casp17.assign_ranked_model(1, "m1", cmd)
        casp17.assign_ranked_model(2, "m1", cmd)
        self.assertEqual(
            casp17.get_ranked_model_names(cmd),
            [None, "m1", None, None, None])

        casp17.assign_ranked_model(2, "m2", cmd)
        self.assertEqual(
            casp17.get_ranked_model_names(cmd),
            [None, "m2", None, None, None])
        self.assertEqual(casp17.get_rank_for_model("m2", cmd), 2)
        self.assertEqual(cmd._casp_get_rank_for_model("m2"), 2)

        with self.assertRaises(casp17.CASPSubmissionError):
            casp17.assign_ranked_model(6, "m3", cmd)

    def testClearRankedModels(self):
        for name in ["m1", "m2"]:
            cmd.fragment("gly", name)

        casp17.reset_ranked_model_names(cmd)
        casp17.assign_ranked_model(1, "m1", cmd)
        casp17.assign_ranked_model(2, "m2", cmd)

        casp17.clear_ranked_model_name("m1", cmd)
        self.assertEqual(
            casp17.get_ranked_model_names(cmd),
            [None, "m2", None, None, None])

        casp17.clear_ranked_model_rank(2, cmd)
        self.assertEqual(casp17.get_ranked_model_names(cmd), [None] * 5)

        with self.assertRaises(casp17.CASPSubmissionError):
            casp17.clear_ranked_model_rank(6, cmd)

    def testRankedModelValidationCleansInvalidObjects(self):
        for name in ["m1", "m2"]:
            cmd.fragment("gly", name)

        casp17.reset_ranked_model_names(cmd)
        casp17.assign_ranked_model(1, "m1", cmd)
        casp17.assign_ranked_model(2, "m2", cmd)
        cmd.delete("m1")

        self.assertEqual(
            casp17.get_ranked_model_names(cmd),
            [None, "m2", None, None, None])
        with self.assertRaises(casp17.CASPSubmissionError):
            casp17.get_validated_ranked_model_names(cmd)

    def testObjectPanelPickStoresPendingModelWithoutQt(self):
        cmd.fragment("gly", "m1")
        self.assertEqual(casp17._casp_object_panel_pick("m1", cmd), "m1")
        self.assertEqual(
            getattr(cmd._pymol, "_casp17_pending_model_name", None), "m1")
        self.assertIsNone(casp17._casp_object_panel_pick("missing", cmd))

    def testBuildCurrentSubmissionUsesRankedOrder(self):
        self.assertTrue(hasattr(cmd, "casp_submit"))

        for name in ["m1", "m2", "m3", "m4", "m5", "m6"]:
            cmd.fragment("gly", name)
        casp17.reset_ranked_model_names(cmd)
        for index, name in enumerate(["m5", "m4", "m3", "m2", "m1"], start=1):
            casp17.assign_ranked_model(index, name, cmd)
        cmd.set("casp_group_id", "test-group")

        with testing.mktemp(".pdb") as filename:
            result = casp17.build_current_submission("T1234", filename, _self=cmd)
            text = open(filename).read()

        self.assertEqual(result["model_names"], ["m5", "m4", "m3", "m2", "m1"])
        self.assertEqual(text.count("MODEL  "), 5)

    def testGroupIDSettingDrivesBuildSubmission(self):
        cmd.set("casp_group_id", "configured-group_2")

        self.load_ranked_models()

        with testing.mktemp(".pdb") as filename:
            result = casp17.build_current_submission("T1234", filename, _self=cmd)
            text = open(filename).read()

        self.assertEqual(casp17.get_group_id(cmd), "configured-group_2")
        self.assertEqual(result["group_id"], "configured-group_2")
        self.assertIn("AUTHOR configured-group_2\n", text)

    def testGroupIDSettingIsRequiredForBuildSubmission(self):
        cmd.set("casp_group_id", "")

        self.load_ranked_models()

        with testing.mktemp(".pdb") as filename:
            with self.assertRaises(casp17.CASPSubmissionError):
                casp17.build_current_submission("T1234", filename, _self=cmd)

    def testPrepareAndRestoreGridPreview(self):
        ranked = self.load_ranked_models()
        cmd.fragment("gly", "m6")
        cmd.disable("m6")
        cmd.set("grid_mode", 0)
        cmd.set("grid_slot", 4, "m1")

        restore_state = casp17.prepare_grid_preview(ranked, cmd)

        self.assertEqual(cmd.get_setting_int("grid_mode"), 1)
        self.assertEqual(
            set(cmd.get_names("public_nongroup_objects", enabled_only=1)),
            set(ranked))
        self.assertEqual(cmd.get_setting_int("grid_slot", "m3"), 3)

        casp17.restore_grid_preview(restore_state, cmd)

        self.assertEqual(cmd.get_setting_int("grid_mode"), 0)
        self.assertNotIn(
            "m6", cmd.get_names("public_nongroup_objects", enabled_only=1))
        self.assertEqual(cmd.get_setting_int("grid_slot", "m1"), 4)

    def testPartialGridPreviewUsesAssignedRankSlots(self):
        for name in ["m1", "m2", "m3"]:
            cmd.fragment("gly", name)
        casp17.reset_ranked_model_names(cmd)
        casp17.assign_ranked_model(2, "m1", cmd)
        casp17.assign_ranked_model(4, "m2", cmd)

        restore_state = casp17.prepare_grid_preview(
            casp17.get_ranked_model_names(cmd), cmd)

        self.assertEqual(cmd.get_setting_int("grid_mode"), 1)
        self.assertEqual(
            set(cmd.get_names("public_nongroup_objects", enabled_only=1)),
            {"m1", "m2"})
        self.assertEqual(cmd.get_setting_int("grid_slot", "m1"), 2)
        self.assertEqual(cmd.get_setting_int("grid_slot", "m2"), 4)
        with self.assertRaises(casp17.CASPSubmissionError):
            casp17.prepare_grid_preview([None] * 5, cmd)

        casp17.restore_grid_preview(restore_state, cmd)

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

        self.assertTrue(hasattr(cmd, "_casp_object_panel_pick"))
        self.assertTrue(hasattr(cmd, "_casp_submission_mode_changed"))
        self.assertTrue(
            hasattr(pymol_qt_gui.PyMOLQtGUI, "open_casp_submission_dialog"))
        self.assertTrue(
            hasattr(pymol_qt_gui.PyMOLQtGUI, "handle_casp_object_panel_pick"))
        self.assertTrue(
            hasattr(pymol_qt_gui.PyMOLQtGUI, "handle_casp_rank_key"))
        self.assertTrue(
            hasattr(pymol_qt_gui.PyMOLQtGUI, "toggle_casp_preview"))
        self.assertTrue(
            hasattr(pymol_qt_gui.PyMOLQtGUI, "clear_casp_rank"))

        from pmg_qt import pymol_gl_widget
        self.assertTrue(
            hasattr(pymol_gl_widget.PyMOLGLWidget, "set_casp_preview_border"))
