# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The ElectronicDensity property."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import List, Tuple

import numpy as np

from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.results import EigenstateResult
from .bases import ElectronicBasis, ElectronicBasisTransform
from .integrals import (
    ElectronicIntegrals,
    IntegralProperty,
    OneBodyElectronicIntegrals,
)
from .particle_number import ParticleNumber


class ElectronicDensity(IntegralProperty):
    """The ElectronicDensity property.

    This is the central property around which the iterative DFT-Embedding is implemented.
    """

    def __init__(self, electronic_integrals: List[ElectronicIntegrals]) -> None:
        """
        Args:
        """
        super().__init__(self .__class__.__name__, electronic_integrals)
        self._norbs = self.get_electronic_integral(ElectronicBasis.MO, 1)._matrices[0].shape[0]
        self._diag_indices: List[int]
        self._aux_ops_offset: int

    def second_q_ops(self) -> List[FermionicOp]:
        """Returns a list containing the Hamiltonian constructed by the stored electronic integrals."""
        aux_ops = []
        aux_ops_b = []
        self._diag_indices = []
        for idx, (mo_i, mo_j) in enumerate(product(range(self._norbs), repeat=2)):
            if mo_i == mo_j:
                self._diag_indices.append(idx)
            alpha_op = FermionicOp(f"+_{mo_i} -_{mo_j}", register_length=2 * self._norbs)
            aux_ops.append(alpha_op)
            beta_op = FermionicOp(
                f"+_{mo_i + self._norbs} -_{mo_j + self._norbs}", register_length=2 * self._norbs
            )
            aux_ops_b.append(beta_op)

        aux_ops.extend(aux_ops_b)
        return aux_ops

    @staticmethod
    def from_particle_number(particle_number: ParticleNumber) -> ElectronicDensity:
        """TODO."""
        return ElectronicDensity(
            [
                OneBodyElectronicIntegrals(
                    ElectronicBasis.MO,
                    (
                        np.diag(particle_number.occupation_alpha),
                        np.diag(particle_number.occupation_beta),
                    ),
                ),
            ]
        )

    def evaluate_particle_number(self, aux_values: np.ndarray) -> Tuple[int, int]:
        """TODO."""
        local_aux_values = aux_values[self._aux_ops_offset :]
        num_a = round(sum([local_aux_values[idx, 0] for idx in self._diag_indices]))
        num_b = round(
            sum([local_aux_values[idx + self._norbs ** 2, 0] for idx in self._diag_indices])
        )
        return (num_a, num_b)

    def transform_basis(self, transform: ElectronicBasisTransform) -> None:
        """Applies an ElectronicBasisTransform to the internal integrals.

        Args:
            transform: the ElectronicBasisTransform to apply.
        """
        for integral in self._electronic_integrals[transform.initial_basis].values():
            self.add_electronic_integral(integral.transform_basis(transform))
            self._norbs = self.get_electronic_integral(ElectronicBasis.MO, 1)._matrices[0].shape[0]
            self._diag_indices = []

    def interpret(self, result: EigenstateResult) -> None:
        """Interprets an :class:`~qiskit_nature.results.EigenstateResult` in this property's context.

        Note that in this specific case, the active density information as evaluated during the
        quantum algorithm gets extracted from the result and is used to update this property itself,
        too.

        Args:
            result: the result to add meaning to.
        """
        aux_op_eigenvalues = result.aux_operator_eigenvalues[0][self._aux_ops_offset :]

        rho_update_a = np.zeros((self._norbs, self._norbs), dtype=float)
        rho_update_b = np.zeros((self._norbs, self._norbs), dtype=float)

        for idx, (mo_i, mo_j) in enumerate(product(range(self._norbs), repeat=2)):
            rho_update_a[mo_i, mo_j] = aux_op_eigenvalues[idx][0].real
            rho_update_b[mo_i, mo_j] = aux_op_eigenvalues[idx + self._norbs ** 2][0].real

        # TODO: implement damping

        self.add_electronic_integral(
            OneBodyElectronicIntegrals(ElectronicBasis.MO, (rho_update_a, rho_update_b))
        )

    def __rmul__(self, other: complex) -> ElectronicDensity:
        return ElectronicDensity([other * int for int in iter(self)])

    def __add__(self, other: ElectronicDensity) -> ElectronicDensity:
        added = deepcopy(self)

        iterator = added.__iter__()
        sum_int = None

        while True:
            try:
                self_int = iterator.send(sum_int)
            except StopIteration:
                break

            sum_int = self_int + other.get_electronic_integral(self_int.basis, 1)

        return added

    def __sub__(self, other: ElectronicDensity) -> ElectronicDensity:
        return self + (-1.0) * other
