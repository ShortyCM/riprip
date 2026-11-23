//! Deterministic corruption cases with independent reference syndromes.

use super::{
	lookup::Entry,
	pcm_frame,
	repair::{
		Geometry,
		Quality,
		Rejection,
		Snapshot,
		unchanged_crcs,
	},
	rs::MAX_PARITY,
};
use crate::KillSwitch;
use std::sync::{
	Arc,
	atomic::AtomicBool,
	LazyLock,
};

static ACTIVE: LazyLock<Arc<AtomicBool>> = LazyLock::new(|| Arc::new(AtomicBool::new(false)));
const SECTOR: usize = 588;

/// # Get Kill Switch.
fn kill_switch() -> KillSwitch { KillSwitch::from(&*ACTIVE) }

/// # Field Product.
///
/// Multiply in the polynomial basis, independently of the decoder's tables.
fn field_product(a: u16, b: u16) -> u16 {
	let mut bits = u32::from(a);
	let mut product = 0;
	for i in 0..16 {
		if b & (1 << i) != 0 { product ^= bits; }
		bits <<= 1;
		if bits & 0x1_0000 != 0 { bits ^= 0x1_100b; }
	}
	u16::try_from(product).expect("reduced field element")
}

/// # Reference Syndrome.
///
/// Build wire-order syndromes by summing columns of the parity check matrix.
fn reference(pcm: &[u16], geometry: Geometry, offset: isize, roots: usize) -> Vec<u16> {
	let lanes = geometry.lanes();
	let first = 2 * geometry.window(offset).start;
	let mut out = vec![0; lanes * roots];
	let mut root = 1;
	for row in out.chunks_exact_mut(lanes) {
		let mut weight = 1;
		for at in (0..geometry.rows).rev() {
			let start = first + at * lanes;
			for (s, &word) in row.iter_mut().zip(&pcm[start..start + lanes]) {
				*s ^= field_product(word, weight);
			}
			root_assert(weight);
			weight = field_product(weight, root);
		}
		root = field_product(root, 2);
	}
	out
}

/// # NonZero Root.
fn root_assert(root: u16) { assert_ne!(root, 0, "syndrome weight must be invertible"); }

/// # Samples to Bytes.
fn bytes(pcm: &[u16]) -> Vec<u8> {
	pcm.as_chunks::<2>().0.iter().flat_map(|&frame| pcm_frame(frame)).collect()
}

/// # CRC32 Wrapper.
fn crc(pcm: &[u16], range: std::ops::Range<usize>) -> u32 {
	crc32fast::hash(&bytes(&pcm[2 * range.start..2 * range.end]))
}

/// # Fixture.
fn fixture(frames: usize) -> Vec<u16> {
	let mut rng = 0x6158_47f0_u32;
	(0..frames * 2).map(|_| {
		rng ^= rng << 13;
		rng ^= rng >> 17;
		rng ^= rng << 5;
		u16::from_le_bytes([rng.to_le_bytes()[0], rng.to_le_bytes()[1]])
	}).collect()
}

/// # Make Entry.
fn record(pcm: &[u16], geometry: Geometry, offset: isize, split: usize) -> Entry {
	let window = geometry.window(offset);
	let pivot = split.checked_add_signed(offset).expect("test boundary in range");
	Entry {
		id: 1,
		confidence: 12,
		disc_crc: crc(pcm, window.clone()),
		checks: vec![
			(0..split, crc(pcm, window.start..pivot)),
			(split..pcm.len() / 2, crc(pcm, pivot..window.end)),
		],
		reference: Vec::new(),
		parity_url: None,
	}
}

#[test]
/// A whole-sector burst straddles a track boundary and one interleave wrap.
fn interleaved_sectors_and_pressing_offsets() {
	let frames = 43 * SECTOR;
	let geometry = Geometry::new(frames, 3 * SECTOR).expect("test geometry");
	let original = fixture(frames);
	for offset in [0, -173, 191] {
		let split = frames / 2;
		let entry = record(&original, geometry, offset, split);
		let parity = reference(&original, geometry, offset, 8);
		let mut damaged = original.clone();
		let mut flags = vec![Quality::Likely; frames];
		let wrap = geometry.window(offset).start + 6 * geometry.stride;
		for frame in wrap - SECTOR / 2..wrap + SECTOR / 2 {
			damaged[2 * frame] ^= 0x430b;
			damaged[2 * frame + 1] ^= 0x9593;
			flags[frame] = Quality::Erasure;
		}
		for frame in split - 93..split + 61 {
			damaged[2 * frame + 1] ^= 0xffff;
			flags[frame] = Quality::Weak;
		}
		damaged[2 * (split + 127)] ^= 0x080d;
		let snapshot = Snapshot::new(geometry, &damaged, &flags, 8, kill_switch())
			.expect("test snapshot");
		let capacity = snapshot.capacity(offset, 8);
		assert_eq!(capacity.max_erasures, 1, "a sector spans distinct codewords");
		assert_eq!(capacity.lanes_over_unknowns, 0, "damage fits across lanes");
		let repair = snapshot.find(&entry, &parity, kill_switch())
			.expect("verified reconstruction");
		assert_eq!(repair.offset, offset, "pressing offset");
		for (word, delta) in repair.corrections { damaged[word] ^= delta; }
		assert!(bytes(&damaged) == bytes(&original), "PCM is byte-for-byte identical");
	}
}

#[test]
fn errors_erasures_and_mixed_capacity_boundaries() {
	let geometry = Geometry::new(83 * 9 + 4, 9).expect("test geometry");
	let frames = 83 * 9 + 4;
	let original = fixture(frames);
	let entry = record(&original, geometry, 0, 344);
	let parity = reference(&original, geometry, 0, MAX_PARITY);
	let start = geometry.window(0).start;

	for (errors, erasures, should_repair) in [
		(1, 0, true), (8, 0, true), (0, 16, true), (4, 8, true),
		(7, 2, true), (2, 12, true), (9, 0, false), (0, 17, false), (5, 7, false),
	] {
		let mut damaged = original.clone();
		let mut flags = vec![Quality::Likely; frames];
		for i in 0..(errors + erasures) {
			let frame = start + (i * 3 + 1) * geometry.stride + 2;
			damaged[frame * 2] ^= 0xb351;
			if i < erasures { flags[frame] = Quality::Erasure; }
		}
		let snapshot = Snapshot::new(
			geometry,
			&damaged,
			&flags,
			MAX_PARITY,
			kill_switch(),
		).expect("test snapshot");
		let capacity = snapshot.capacity(0, MAX_PARITY);
		assert_eq!(capacity.max_erasures, erasures, "same-lane concentration");
		let repair = snapshot.find(&entry, &parity, kill_switch());
		assert_eq!(
			repair.is_ok(),
			should_repair,
			"{errors} errors and {erasures} erasures",
		);
		if let Ok(repair) = repair {
			for (word, delta) in repair.corrections { damaged[word] ^= delta; }
			assert!(
				bytes(&damaged) == bytes(&original),
				"PCM is byte-for-byte identical",
			);
		}
	}
}

#[test]
fn checksum_and_protected_samples_gate_acceptance() {
	let frames = 503;
	let geometry = Geometry::new(frames, 26).expect("test geometry");
	let original = fixture(frames);
	let parity = reference(&original, geometry, 0, 8);
	let mut damaged = original.clone();
	damaged[105] ^= 1;
	damaged[1] ^= 0x2a; // Outside the protected part of the audio image.
	let mut flags = vec![Quality::Likely; frames];
	flags[0] = Quality::Erasure;
	flags[52] = Quality::Trusted;
	let mut entry = record(&original, geometry, 0, 251);
	let snapshot = Snapshot::new(geometry, &damaged, &flags, 8, kill_switch())
		.expect("test snapshot");
	assert!(
		snapshot.find(&entry, &parity, kill_switch()).is_err(),
		"preserve verified PCM",
	);
	flags[52] = Quality::Erasure;
	let snapshot = Snapshot::new(geometry, &damaged, &flags, 8, kill_switch())
		.expect("test snapshot");
	assert_eq!(snapshot.capacity(0, 8).outside, 1, "report an uncovered drive error");
	entry.disc_crc ^= 1;
	assert_eq!(
		snapshot.find(&entry, &parity, kill_switch()).err(),
		Some(Rejection::Checksum),
		"reject a wrong disc CRC",
	);
	entry.disc_crc ^= 1;
	entry.checks[0].1 ^= 1;
	assert_eq!(
		snapshot.find(&entry, &parity, kill_switch()).err(),
		Some(Rejection::Checksum),
		"reject a wrong track CRC",
	);
	entry.checks[0].1 ^= 1;
	let repair = snapshot.find(&entry, &parity, kill_switch()).expect("correct CRCs");
	assert_eq!(repair.corrections, [(105, 1)], "do not change unprotected PCM");
}

#[test]
fn false_erasure_hints_do_not_block_a_verified_error_only_repair() {
	let frames = 609;
	let geometry = Geometry::new(frames, 31).unwrap();
	let original = fixture(frames);
	let entry = record(&original, geometry, 0, 300);
	let parity = reference(&original, geometry, 0, 4);
	let mut damaged = original.clone();
	let mut flags = vec![Quality::Likely; frames];

	// A multiple of the generator has zero syndrome. Splitting its five
	// consecutive nonzero coefficients between genuine errors and another
	// plausible patch makes two syndrome-valid decodes.
	let mut generator = [1, 0, 0, 0, 0];
	let mut root = 1;
	for degree in 1..=4 {
		for i in (1..=degree).rev() {
			generator[i] ^= field_product(generator[i - 1], root);
		}
		root = field_product(root, 2);
	}
	let lane_frame = |row: usize| geometry.window(0).start + (3 + row) * geometry.stride + 9;
	damaged[lane_frame(0) * 2] ^= generator[0];
	damaged[lane_frame(1) * 2] ^= generator[1];
	flags[lane_frame(2)] = Quality::Erasure;
	flags[lane_frame(3)] = Quality::Erasure;
	let snapshot = Snapshot::new(geometry, &damaged, &flags, 4, kill_switch()).unwrap();
	let repair = snapshot.find(&entry, &parity, kill_switch())
		.expect("CRC selects the actual repair");
	for (at, delta) in repair.corrections { damaged[at] ^= delta; }
	assert_eq!(
		bytes(&damaged),
		bytes(&original),
		"the right byte values survive misleading C2",
	);
}

#[test]
fn shifted_crc_scan_matches_independent_pcm_slices() {
	let frames = 780;
	let geometry = Geometry::new(frames, 39).expect("test geometry");
	let pcm = fixture(frames);
	for (offset, checksum) in unchanged_crcs(&pcm, geometry, kill_switch()) {
		assert_eq!(checksum, crc(&pcm, geometry.window(offset)), "offset {offset}");
	}
}

#[test]
fn verified_checkpoint_keeps_only_reachable_guards_and_htoa_pending() {
	use super::{Audio, track_coverage};
	use crate::{RipOptions, ReadOffset, rip::RipEntry};
	let toc = cdtoc::Toc::from_cdtoc("2+326+35F+396").unwrap();
	let opts = RipOptions::default().with_offset(ReadOffset::try_from(6).unwrap());
	assert!(toc.htoa().is_some(), "the fixture has audio before track one");
	let first = toc.audio_leadin_normalized().cast_signed() * 588;
	let length = usize::try_from(toc.duration().samples()).unwrap();
	let split = usize::try_from(toc.audio_tracks().next().unwrap().duration().samples()).unwrap();
	let mut audio = Audio {
		pcm: Vec::new(), quality: vec![Quality::Likely; length],
		boundaries: vec![0, split, length], missing: 0, overlap_changes: Vec::new(),
	};
	audio.quality[0] = Quality::Weak;
	audio.quality[split / 2] = Quality::Erasure;
	audio.quality[length - 1] = Quality::Erasure;
	let window = 10..length - 10;
	let mut coverage = track_coverage(&toc, &opts, &audio, window.clone());
	let mut first_track = RipEntry::new(&toc, 1, 0).unwrap();
	first_track.dst = Some("01.wav".into());
	first_track.ctdb_coverage = coverage.remove(&1);
	let guard = first_track.ctdb_coverage.as_ref().unwrap();
	assert!(guard.needs_sample(first), "keep the reachable leading guard");
	assert!(! guard.needs_sample(first + 10), "skip the verified audio at the boundary");
	assert!(! first_track.skippable(), "an uncertain reachable guard still needs a pass");

	let mut last_track = RipEntry::new(&toc, 2, 0).unwrap();
	last_track.dst = Some("02.wav".into());
	last_track.ctdb_coverage = coverage.remove(&2);
	assert!(
		last_track.skippable(),
		"an offset-inaccessible tail cannot benefit from more passes",
	);
	let htoa = RipEntry::new(&toc, 0, 0).unwrap();
	assert!(coverage.is_empty() && !htoa.skippable(), "CTDB adds no coverage for HTOA");

	audio.quality[0] = Quality::Likely;
	first_track.ctdb_coverage = track_coverage(&toc, &opts, &audio, window).remove(&1);
	assert!(first_track.skippable(), "another pass recovered the final reachable guard");
}

#[test]
fn rejects_crc_valid_patches_at_ambiguous_offsets() {
	let geometry = Geometry::new(260, 10).unwrap();
	let mut original = vec![0; 520];
	original[211] = 0x7e55;
	let damaged = vec![0; 520];
	let quality = vec![Quality::Likely; 260];
	let entry = record(&original, geometry, 0, 130);
	let parity = reference(&original, geometry, 0, 4);
	let snapshot = Snapshot::new(geometry, &damaged, &quality, 4, kill_switch())
		.expect("test snapshot");
	assert_eq!(
		snapshot.find(&entry, &parity, kill_switch()).err(),
		Some(Rejection::Ambiguous),
		"the same pulse can be synthesized at different positions relative to the local rip",
	);
}
