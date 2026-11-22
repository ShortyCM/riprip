/*!
# Rip Rip Hooray: Reed–Solomon decoding over CTDB's GF(2^16).
*/

/// # Number of Nonzero Field Elements.
const ORDER: usize = u16::MAX as usize;

/// # CTDB's Field Polynomial (w/ Leading Term).
const POLYNOMIAL: u32 = 0x1_100b;

/// # Largest Syndrome Record Supported by CTDB.
pub(super) const MAX_PARITY: usize = 16;

#[expect(clippy::cast_possible_truncation, reason = "False positive.")]
/// # Powers of Two.
///
/// Note these are duplicated to allow lookup without reduction.
static EXP: [u16; 2 * ORDER + 1] = {
	let mut exp = [0_u16; 2 * ORDER + 1];
	let mut value: u32 = 1;
	let mut power = 0;
	while power < ORDER {
		assert!(value <= u16::MAX as u32, "BUG: value would truncate!");
		exp[power] = value as u16;
		exp[power + ORDER] = value as u16;

		power += 1;
		value <<= 1;
		if value & 0x1_0000 != 0 { value ^= POLYNOMIAL; }
	}
	exp
};

#[expect(clippy::cast_possible_truncation, reason = "False positive.")]
/// # Discrete Base-2 Log.
static LOG: [u16; ORDER + 1] = {
	let mut log = [0_u16; ORDER + 1];
	let mut value: u32 = 1;
	let mut power = 0;
	while power < ORDER {
		assert!(power <= u16::MAX as usize, "BUG: power would truncate!");
		log[value as usize] = power as u16;

		power += 1;
		value <<= 1;
		if value & 0x1_0000 != 0 { value ^= POLYNOMIAL; }
	}
	log
};

/// # Field Arithmetic With Primitive Element 2.
pub(super) struct Field;

impl Field {
	/// # Evaluate the Primitive Element to the Given Power.
	pub(super) const fn alpha(power: usize) -> u16 { EXP[power % ORDER] }

	/// # Multiply by a Power of the Primitive Element.
	pub(super) const fn mul_alpha(value: u16, power: usize) -> u16 {
		if value == 0 { 0 }
		else { EXP[LOG[value as usize] as usize + power % ORDER] }
	}

	/// # Multiply Field Elements.
	const fn mul(a: u16, b: u16) -> u16 {
		if a == 0 || b == 0 { 0 }
		else { EXP[LOG[a as usize] as usize + LOG[b as usize] as usize] }
	}

	/// # Divide by a Nonzero Field Element.
	const fn div(a: u16, b: u16) -> u16 {
		debug_assert!(b != 0, "Bug: field division by zero.");

		if a == 0 || b == 0 { 0 }
		else { EXP[ORDER + LOG[a as usize] as usize - LOG[b as usize] as usize] }
	}

	#[expect(clippy::cast_possible_truncation, reason = "False positive.")]
	/// # Fixed Field Multiplication For 2-Byte Lookups.
	pub(super) const fn multiplier(factor: u16) -> [[u16; 256]; 2] {
		let mut tables = [[0_u16; 256]; 2];
		let mut byte = 0;
		while byte < 2 {
			let mut value = 0_usize;
			while value < 256 {
				tables[byte][value] = Self::mul(factor, (value as u16) << (byte * 8));
				value += 1;
			}
			byte += 1;
		}
		tables
	}

	/// # Decode.
	///
	/// Find errors in a shortened codeword whose first word has degree
	/// `words - 1`.
	///
	/// Returns position and XOR magnitude for the changed words, or `None` on
	/// failure.
	pub(super) fn decode(
		syndromes: &[u16],
		words: usize,
		erasures: &[usize],
	) -> Option<Vec<(usize, u16)>> {
		// Short circuit: remind compiler that words fits u16.
		if ORDER < words { return None; }

		// Short circuit: bail if the the syndrome is empty or all zero.
		if syndromes.is_empty() || syndromes.iter().all(|&s| s == 0) {
			return Some(Vec::new());
		}

		// Short circuit: bail if there are too many erasures or syndromes.
		if syndromes.len() < erasures.len() || MAX_PARITY < syndromes.len() {
			return None;
		}

		// For each known location X, taking S[j+1] + X*S[j] cancels
		// its contribution and leaves a shorter sequence for the unknowns.
		let mut reduced = [0; MAX_PARITY];
		let mut remaining = syndromes.len();
		reduced[..remaining].copy_from_slice(syndromes);
		for &position in erasures {
			if position >= words { return None; }
			let x = Self::alpha(words - 1 - position);
			for j in 0..remaining - 1 {
				reduced[j] = reduced[j + 1] ^ Self::mul(x, reduced[j]);
			}
			remaining -= 1;
		}

		let mut positions = erasures.to_vec();
		let (locator, degree) = Self::recurrence(&reduced[..remaining]);
		if 2 * degree > remaining || locator[degree] == 0 { return None; }
		if degree == 1 {
			let power = LOG[usize::from(locator[1])] as usize;
			positions.push(words.checked_sub(1 + power)?);
		}
		else if degree > 1 {
			if ! Self::splits(&locator[..=degree]) { return None; }
			for power in 0..words {
				let inverse = Self::alpha(ORDER - power);
				let value = locator[..=degree].iter().rev().fold(0, |sum, &term| {
					Self::mul(sum, inverse) ^ term
				});
				if value == 0 {
					positions.push(words - 1 - power);
					if positions.len() == erasures.len() + degree { break; }
				}
			}
		}
		positions.sort_unstable();

		// Short circuit: bail if the positions exceed the degree or overlap.
		if
			positions.len() != erasures.len() + degree ||
			positions.windows(2).any(|pair| pair[0] == pair[1])
		{
			return None;
		}

		// With the locations fixed, S[j] = sum(magnitude[i] * X[i]^j).
		// Solve the small Vandermonde system and check every syndrome.
		let mut matrix = [[0; MAX_PARITY + 1]; MAX_PARITY];
		let count = positions.len();
		for (row, equation) in matrix[..count].iter_mut().enumerate() {
			for (col, &position) in positions.iter().enumerate() {
				equation[col] = Self::alpha(row * (words - 1 - position));
			}
			equation[count] = syndromes[row];
		}
		for pivot in 0..count {
			let row = (pivot..count).find(|&row| matrix[row][pivot] != 0)?;
			matrix.swap(pivot, row);
			let divisor = matrix[pivot][pivot];
			for value in &mut matrix[pivot][pivot..=count] {
				*value = Self::div(*value, divisor);
			}
			let normalized = matrix[pivot];
			for (row, equation) in matrix[..count].iter_mut().enumerate() {
				if row == pivot { continue; }
				let factor = equation[pivot];
				for col in pivot..=count {
					equation[col] ^= Self::mul(factor, normalized[col]);
				}
			}
		}

		// Find the changes and return them if expected.
		let changes: Vec<_> = positions.into_iter()
			.zip(matrix.iter().map(|row| row[count]))
			.filter(|&(_, magnitude)| magnitude != 0)
			.collect();
		for (root, &expected) in syndromes.iter().enumerate() {
			let actual = changes.iter().fold(0, |sum, &(position, magnitude)| {
				sum ^ Self::mul(magnitude, Self::alpha(root * (words - 1 - position)))
			});
			if actual != expected { return None; }
		}
		Some(changes)
	}

	/// # Splits.
	///
	/// Every valid locator divides `x^(2^16) + x`. Check this invariant before
	/// scanning thousands of possible positions for a wrong pressing alignment
	/// (while also proving each root is unique).
	fn splits(locator: &[u16]) -> bool {
		let degree = locator.len() - 1;
		let mut polynomial = [0; MAX_PARITY];
		polynomial[1] = 1;
		let mut modulus = [0; MAX_PARITY];
		for (out, &term) in modulus[..degree].iter_mut().zip(locator) {
			*out = Self::div(term, locator[degree]);
		}
		for _ in 0..16 {
			let mut square = [0; 2 * MAX_PARITY];
			for (i, &value) in polynomial[..degree].iter().enumerate() {
				square[2 * i] = Self::mul(value, value);
			}
			for i in (degree..2 * degree - 1).rev() {
				let factor = square[i];
				for (term, &coefficient) in square[i - degree..i].iter_mut().zip(&modulus) {
					*term ^= Self::mul(factor, coefficient);
				}
			}
			polynomial[..degree].copy_from_slice(&square[..degree]);
		}
		polynomial[1] ^= 1;
		polynomial[..degree].iter().all(|&term| term == 0)
	}

	/// # Berlekamp–Massey Shortest Recurrence.
	///
	/// Find and return the shortest recurrence for a sequence of field elements.
	fn recurrence(sequence: &[u16]) -> ([u16; MAX_PARITY + 1], usize) {
		let mut current = [0; MAX_PARITY + 1];
		current[0] = 1;
		let mut previous = current;
		let mut degree = 0;
		let mut distance = 1;
		let mut last = 1;
		for (n, &symbol) in sequence.iter().enumerate() {
			let discrepancy = (1..=degree).fold(symbol, |sum, i| {
				sum ^ Self::mul(current[i], sequence[n - i])
			});
			if discrepancy == 0 {
				distance += 1;
				continue;
			}

			let before = current;
			let scale = Self::div(discrepancy, last);
			for (term, &prior) in current[distance..].iter_mut().zip(previous.iter()) {
				*term ^= Self::mul(scale, prior);
			}
			if degree * 2 <= n {
				previous = before;
				degree = n + 1 - degree;
				last = discrepancy;
				distance = 1;
			}
			else { distance += 1; }
		}
		(current, degree)
	}
}
