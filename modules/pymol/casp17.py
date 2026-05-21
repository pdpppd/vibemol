from __future__ import annotations

import os
import re
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pymol

cmd = sys.modules["pymol.cmd"]


DEFAULT_GROUP_ID = ""
SUBMIT_URL = "https://predictioncenter.org/casp17/submit"

AA3 = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


class CASPSubmissionError(Exception):
    pass


def _normalize_target_id(target_id):
    target_id = str(target_id).strip().upper()
    if not re.fullmatch(r"[A-Z][0-9]{4}", target_id):
        raise CASPSubmissionError(
            "CASP target ID must look like H1312 or T1234"
        )
    return target_id


def _require_email(email):
    email = str(email).strip()
    if not email:
        raise CASPSubmissionError("CASP registered email address is required")
    return email


def _normalize_group_id(group_id):
    group_id = str(group_id).strip()
    if not group_id:
        raise CASPSubmissionError("CASP group ID is required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", group_id):
        raise CASPSubmissionError(
            "CASP group ID may only contain letters, numbers, '.', '_', and '-'"
        )
    return group_id


def get_group_id(_self=cmd, required=False):
    try:
        group_id = _self.get_setting_text("casp_group_id").strip()
    except Exception:
        group_id = DEFAULT_GROUP_ID
    if required or group_id:
        return _normalize_group_id(group_id)
    return group_id


def set_group_id(group_id, _self=cmd):
    group_id = _normalize_group_id(group_id)
    _self.set("casp_group_id", group_id, log=1, quiet=0)
    return group_id


def is_atom_record(line):
    return line.startswith("ATOM  ") or line.startswith("HETATM")


def chain_id(line):
    return line[21:22]


def atom_name(line):
    return line[12:16].strip()


def residue_name(line):
    return line[17:20].strip()


def residue_id(line):
    return line[22:27]


def chain_order(lines):
    order = []
    seen = set()
    for line in lines:
        if is_atom_record(line):
            chain = chain_id(line)
            if chain not in seen:
                seen.add(chain)
                order.append(chain)
    return order


def chain_sequences(lines):
    residues = {}
    seen_residues = {}
    for line in lines:
        if not is_atom_record(line):
            continue
        resname = residue_name(line)
        if atom_name(line) != "CA" or resname not in AA3:
            continue
        chain = chain_id(line)
        resid = residue_id(line)
        seen_residues.setdefault(chain, set())
        if resid in seen_residues[chain]:
            continue
        seen_residues[chain].add(resid)
        residues.setdefault(chain, []).append((resid, resname))
    return {chain: tuple(seq) for chain, seq in residues.items()}


def parent_chains(lines):
    sequences = chain_sequences(lines)
    unique_sequences = set()
    parents = set()

    for chain in chain_order(lines):
        seq = sequences.get(chain)
        if not seq:
            parents.add(chain)
            continue
        if seq not in unique_sequences:
            unique_sequences.add(seq)
            parents.add(chain)

    return parents


def normalized_atom_line(line):
    if len(line) > 80:
        return line[:80]
    return line.ljust(80)


def write_model(out, model_index, lines):
    chains_needing_parent = parent_chains(lines)
    current_chain = None
    emitted_parent = set()
    last_output_was_ter = False

    print("MODEL  %d" % model_index, file=out)

    for line in lines:
        if is_atom_record(line):
            chain = chain_id(line)
            if chain != current_chain:
                if current_chain is not None and not last_output_was_ter:
                    print("TER", file=out)
                if chain in chains_needing_parent and chain not in emitted_parent:
                    print("PARENT N/A", file=out)
                    emitted_parent.add(chain)
                current_chain = chain
            print(normalized_atom_line(line), file=out)
            last_output_was_ter = False
        elif line.startswith("TER"):
            print("TER", file=out)
            last_output_was_ter = True

    if current_chain is not None and not last_output_was_ter:
        print("TER", file=out)
    print("END", file=out)


def get_ranked_model_names(_self=cmd):
    names = []
    for name in _self.get_names("public_nongroup_objects"):
        if _self.get_type(name) == "object:molecule":
            names.append(name)
            if len(names) == 5:
                break
    return names


def validate_model_names(model_names):
    if len(model_names) < 5:
        raise CASPSubmissionError(
            "CASP17 submission mode requires at least five public molecule objects"
        )
    return list(model_names[:5])


def get_model_pdb_lines(model_name, _self=cmd):
    pdb = _self.get_pdbstr(model_name, state=-1, quiet=1)
    lines = pdb.splitlines()
    if not any(is_atom_record(line) for line in lines):
        raise CASPSubmissionError(
            'Object "%s" did not export any ATOM/HETATM records' % model_name
        )
    return lines


def write_submission(model_lines, target_id, output_path, group_id=None):
    target_id = _normalize_target_id(target_id)
    if group_id is None:
        group_id = DEFAULT_GROUP_ID
    group_id = _normalize_group_id(group_id)
    model_lines = list(model_lines)
    if len(model_lines) > 5:
        raise CASPSubmissionError(
            "CASP TS submissions may contain at most five ranked models"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as out:
        print("PFRMAT TS", file=out)
        print("TARGET %s" % target_id, file=out)
        print("AUTHOR %s" % group_id, file=out)
        print(
            "METHOD Coordinates assembled from provided ranked model files.",
            file=out,
        )
        print(
            "METHOD PARENT N/A used for modeled subunits; per-atom confidence retained in B-factor field.",
            file=out,
        )
        for index, lines in enumerate(model_lines, start=1):
            write_model(out, index, lines)

    return output_path


def build_current_submission(target_id, output_path=None, model_names=None,
                             group_id=None, _self=cmd):
    target_id = _normalize_target_id(target_id)
    if group_id is None:
        group_id = get_group_id(_self, required=True)
    group_id = _normalize_group_id(group_id)
    if model_names is None:
        model_names = get_ranked_model_names(_self)
    model_names = validate_model_names(model_names)

    if output_path is None:
        output_path = Path(os.getcwd()) / ("%s_TS_%s.pdb" % (target_id, group_id))

    model_lines = [get_model_pdb_lines(name, _self) for name in model_names]
    output_path = write_submission(model_lines, target_id, output_path, group_id)
    return {
        "group_id": group_id,
        "model_names": model_names,
        "output_path": output_path,
    }


def _multipart_form_data(fields, files):
    boundary = "PyMOLCASP17%s" % uuid.uuid4().hex
    chunks = []

    for name, value in fields.items():
        chunks.extend([
            "--%s\r\n" % boundary,
            'Content-Disposition: form-data; name="%s"\r\n\r\n' % name,
            str(value),
            "\r\n",
        ])

    for name, path in files.items():
        path = Path(path)
        chunks.extend([
            "--%s\r\n" % boundary,
            (
                'Content-Disposition: form-data; name="%s"; filename="%s"\r\n'
                % (name, path.name)
            ),
            "Content-Type: chemical/x-pdb\r\n\r\n",
            path.read_bytes(),
            "\r\n",
        ])

    chunks.append("--%s--\r\n" % boundary)

    body = bytearray()
    for chunk in chunks:
        if isinstance(chunk, bytes):
            body.extend(chunk)
        else:
            body.extend(chunk.encode("utf-8"))

    return bytes(body), "multipart/form-data; boundary=%s" % boundary


def submit_to_casp(output_path, email, url=SUBMIT_URL, opener=None, timeout=60):
    email = _require_email(email)
    body, content_type = _multipart_form_data(
        {"email": email},
        {"prediction_file": output_path},
    )
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
            "User-Agent": "PyMOL CASP17 Submission Wizard",
        },
    )

    try:
        if opener is None:
            response = urllib.request.urlopen(request, timeout=timeout)
        else:
            try:
                response = opener.open(request, timeout=timeout)
            except TypeError:
                response = opener.open(request)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace").strip()
        message = "CASP submission failed with HTTP %s" % e.code
        if detail:
            message += ": " + detail
        raise CASPSubmissionError(message) from None
    except urllib.error.URLError as e:
        raise CASPSubmissionError("CASP submission failed: %s" % e.reason) from None

    try:
        status = getattr(response, "status", getattr(response, "code", 200))
        response_text = response.read().decode("utf-8", "replace").strip()
    finally:
        close = getattr(response, "close", None)
        if close is not None:
            close()

    if not 200 <= int(status) < 300:
        message = "CASP submission failed with HTTP %s" % status
        if response_text:
            message += ": " + response_text
        raise CASPSubmissionError(message)

    return response_text


def prepare_grid_preview(model_names, _self=cmd):
    model_names = list(model_names)
    all_objects = _self.get_names("public_nongroup_objects")
    enabled_objects = set(_self.get_names("public_nongroup_objects", enabled_only=1))
    grid_slots = {}
    for name in all_objects:
        try:
            grid_slots[name] = _self.get_setting_int("grid_slot", name)
        except Exception:
            pass

    restore_state = {
        "grid_mode": _self.get_setting_int("grid_mode"),
        "enabled_objects": enabled_objects,
        "all_objects": list(all_objects),
        "grid_slots": grid_slots,
    }

    _self.set("grid_mode", 1, quiet=1)
    for index, name in enumerate(model_names, start=1):
        _self.set("grid_slot", index, name, quiet=1)
        _self.enable(name)

    submitted = set(model_names)
    for name in all_objects:
        if name not in submitted:
            _self.disable(name)

    _self.zoom(" ".join(model_names), animate=0)
    return restore_state


def restore_grid_preview(restore_state, _self=cmd):
    for name, value in restore_state.get("grid_slots", {}).items():
        if name in _self.get_names("all"):
            _self.set("grid_slot", value, name, quiet=1)

    enabled_objects = restore_state.get("enabled_objects", set())
    for name in restore_state.get("all_objects", []):
        if name not in _self.get_names("all"):
            continue
        if name in enabled_objects:
            _self.enable(name)
        else:
            _self.disable(name)

    _self.set("grid_mode", restore_state.get("grid_mode", 0), quiet=1)


def casp_submit(email, target_id, group_id=None, quiet=1, _self=cmd):
    """
DESCRIPTION

    Submit the top five public molecule objects in the object panel to CASP17.

USAGE

    casp_submit email, target_id [, group_id]

ARGUMENTS

    email = registered CASP email address
    target_id = CASP target ID, for example H1312 or T1234
    group_id = optional CASP group ID. If omitted, uses casp_group_id,
    which must be configured first.
    """
    try:
        result = build_current_submission(target_id, group_id=group_id, _self=_self)
        response_text = submit_to_casp(result["output_path"], email)
    except CASPSubmissionError as e:
        raise pymol.CmdException(str(e)) from None

    result["response_text"] = response_text
    if not int(quiet):
        print(" CASP17 submission written to %s" % result["output_path"])
        if response_text:
            print(response_text)
    return result
