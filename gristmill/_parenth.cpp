/** Parenthesization of tensor contractions.
 *
 * This file contains the core dynamic programming + branch and bound solution
 * to the tensor parenthesization problem, implemented in C++ for efficiency.
 */

#include <Python.h>

#include <cassert>
#include <limits>
#include <vector>

#include <cpypp.hpp>
#include <fbitset.hpp>
#include <libparenth.hpp>

using namespace cpypp;
using namespace fbitset;
using namespace libparenth;

//
// StructSequence types for the results
// ====================================
//

// clang-format off

static constexpr Py_ssize_t INTERM_SUMS = 0;
static constexpr Py_ssize_t INTERM_EXTS = 1;
static constexpr Py_ssize_t INTERM_EVALS = 2;
static constexpr Py_ssize_t INTERM_N_FIELDS = 3;

static PyStructSequence_Field interm_fields[] = {
    {"sums", "The summations in the intermediate"},
    {"exts", "The external indices to the intermediate"},
    {"evals", "The evaluation methods"},
    {NULL, NULL}
};

static PyStructSequence_Desc interm_desc = {
    "Interm",
    "Information about an intermediate",
    interm_fields,
    INTERM_N_FIELDS
};

static constexpr Py_ssize_t EVAL_OPS = 0;
static constexpr Py_ssize_t EVAL_SUMS = 1;
static constexpr Py_ssize_t EVAL_COST = 2;
static constexpr Py_ssize_t EVAL_N_FIELDS = 3;

static PyStructSequence_Field eval_fields[] = {
    {"ops", "The factors in the left/right operands"},
    {"sums", "The summations for the last step"},
    {"cost", "The aggregate cost of the evaluation"},
    {NULL, NULL}
};

static PyStructSequence_Desc eval_desc = {
    "Eval",
    "An evaluation for an intermediate",
    eval_fields,
    EVAL_N_FIELDS
};

// clang-format on

static Static_type interm_type{};

static Static_type eval_type{};

//
// Core function
// =============
//

using Limb = unsigned long long;

using Ext_limbs = std::vector<Limb>;

constexpr auto LIMB_BITS = std::numeric_limits<Limb>::digits;

using Dims = std::vector<Handle>;

using Factor = std::vector<Size>;

using Factors = std::vector<Factor>;

/** Builds a tuple for the indices of the set bits.
 *
 * The bits should be given as an Fbitset instantiantion.
 */

template <typename T> static inline Handle build_idx_tuple(const T& bits)
{
    auto size = bits.count();
    Tuple res(size);

    Py_ssize_t curr = 0;
    for (auto i = bits.begin(); i; ++i, ++curr) {
        res.setitem(curr, Handle(long{ *i }));
    }

    assert(curr == size);
    return res;
}

/** Builds an entry for an intermediate.
 *
 * The information about the intermediate should be given as the entry for the
 * intermediate in the main memoir from libparenth.
 */

template <typename T> static inline Handle build_interm(T& info)
{
    auto n_evals = info.evals.size();
    Tuple evals(n_evals);
    for (size_t i = 0; i < n_evals; ++i) {
        const auto& eval = info.evals[i];

        assert(eval_type.is_ready());
        Struct_sequence curr(eval_type);

        Tuple ops(2);
        ops.setitem(0, build_idx_tuple(eval.ops.first));
        ops.setitem(1, build_idx_tuple(eval.ops.second));

        curr.setitem(EVAL_OPS, std::move(ops));
        curr.setitem(EVAL_SUMS, build_idx_tuple(eval.sums));
        curr.setitem(EVAL_COST, std::move(eval.cost));

        evals.setitem(i, std::move(curr));
    }

    assert(interm_type.is_ready());
    Struct_sequence res(interm_type);

    res.setitem(INTERM_SUMS, build_idx_tuple(info.sums));
    res.setitem(INTERM_EXTS, build_idx_tuple(info.exts));
    res.setitem(INTERM_EVALS, std::move(evals));

    return res;
}

/** The core parenthesization function.
 *
 * This function is responsible for invoking the libparenth core function and
 * translate the result back into Python objects.  It can be instantiated with
 * different types to be used for the subsets of dimensions and factors.
 */

template <typename DS, typename FS>
static inline Handle parenth_core(const Dims& dims, size_t n_sums,
    const Factors& factors, Mode mode, bool if_incl)
{
    Parenther<Handle, DS, FS> parenther(
        dims.begin(), dims.end(), n_sums, factors.begin(), factors.end());

    auto opt_res = parenther.opt(mode, if_incl);

    Handle res(PyDict_New());
    for (const auto& interm : opt_res) {
        const auto& factors = interm.first;
        const auto& info = interm.second;

        Handle key = build_idx_tuple(factors);
        Handle val = build_interm(info);

        if (PyDict_SetItem(res.get(), key.get(), val.get()) < 0) {
            throw Exc_set{};
        }
    }

    return res;
}

//
// Python interface main function
// ==============================
//

/** Docstring for the core function.
 */

static const char* parenth_docstring = R"__doc__(
Find parenthesization of tensor contraction.

The optimal parenthesization of the given contraction will be attempted to be
found by a strategy combining dynamic programming and branch-and-bound
technique.

Parameters
----------

dims:
    The sizes of all the dimensions in the problem in an iterable.

n_sums:
    The number of summations in the problem.  The first ``n_sums`` of the
    dimensions are going to be taked as summations, with the later ones taken
    as external indices.  Note that the summations needs to be already sorted
    in terms of their size.

factors:
    The factors in the problem, given as an iterable, which gives an iterable
    for the index of the dimensions involved by the factor.

mode : int
    The mode for termination of the main loop, 0 for greedy mode, 1 for the
    normal mode, and 2 for the exhaustive mode.

if_inclusive : bool
    If the traversed non-optimal evaluations are to be stored as well.

Return
------

The optimal cost of the contraction.

)__doc__";

/** The main Python interface function.
 */

static PyObject* parenth_func(PyObject* self, PyObject* args)
{
    Handle dims_inp{};
    Handle factors_inp{};
    int n_sums;
    int mode_inp;
    int if_incl;

    try {
        if (!PyArg_ParseTuple(args, "OiOip", dims_inp.read(), &n_sums,
                factors_inp.read(), &mode_inp, &if_incl))
            throw Exc_set{};

        std::vector<Handle> dims(dims_inp.begin(), dims_inp.end());
        std::vector<std::vector<Size>> factors{};
        for (auto& factor : factors_inp) {
            factors.emplace_back();
            for (auto& i : factor) {
                factors.back().push_back(i.as<long>());
            }
        }

        Mode mode;
        switch (mode_inp) {
        case 0:
            mode = Mode::GREEDY;
            break;
        case 1:
            mode = Mode::NORMAL;
            break;
        case 2:
            mode = Mode::EXHAUST;
            break;
        default:
            assert(0);
        }

        auto n_dims = dims.size();
        auto n_factors = factors.size();

        Handle res{};

        // Dispatch to different instantiation of the same template for maximum
        // performance.  For problems that fits into a small size, disabling
        // the usage of external limb containers could boost the performance by
        // quite a bit with much less branches, more loop unrolling, and
        // slightly less space consumption.

        if (n_dims < LIMB_BITS && n_factors < LIMB_BITS) {
            res = parenth_core<Fbitset<1, Limb, No_ext>,
                Fbitset<1, Limb, No_ext>>(dims, n_sums, factors, mode, if_incl);
        } else if (n_dims < LIMB_BITS && n_factors < 2 * LIMB_BITS) {
            res = parenth_core<Fbitset<1, Limb, No_ext>,
                Fbitset<2, Limb, No_ext>>(dims, n_sums, factors, mode, if_incl);
        } else {
            // Final fall back mode.
            res = parenth_core<Fbitset<1, Limb, Ext_limbs>,
                Fbitset<2, Limb, Ext_limbs>>(
                dims, n_sums, factors, mode, if_incl);
        }

        return res.release();
    } catch (cpypp::Exc_set) {
        return NULL;
    }
}

//
// Python module definition
// ========================
//

/** Docstring for the module.
 */

static const char* module_docstring = R"__doc__(
Core function for parenthesization of contractions.
)__doc__";

/** Methods in the module.
 */

// clang-format off

static PyMethodDef module_methods[] = {
    { "parenth", (PyCFunction)parenth_func, METH_VARARGS, parenth_docstring },
    { NULL, NULL, 0, NULL }
};

// clang-format on

/** Executes the initialization of the module.
 *
 * The types for the results are added to be module.
 */

static int module_exec(PyObject* m)
{
    try {
        interm_type.make_ready(interm_desc);
        eval_type.make_ready(eval_desc);

        Module module(m);

        module.add_object("Interm", interm_type.get_handle());
        module.add_object("Eval", eval_type.get_handle());
    } catch (Exc_set) {
        return -1;
    }
    return 0;
}

/** Slots for the _parenth module definition.
 */

static struct PyModuleDef_Slot module_slots[] = {
    { Py_mod_exec, (void*)module_exec },
    { 0, NULL },
};

/** _parenth core module definition.
 */

// clang-format off

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "gristmill._parenth",
    module_docstring,
    0, // m-size
    module_methods,
    module_slots,
    NULL, // Transverse
    NULL, // Clear
    NULL  // Free
};

// clang-format on

/** The published function.
 */

PyMODINIT_FUNC PyInit__parenth(void) { return PyModuleDef_Init(&module_def); }
