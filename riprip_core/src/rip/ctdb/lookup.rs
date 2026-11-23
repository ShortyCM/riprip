//! CTDB v3 lookup and syndrome records.

use crate::{
	cache_path,
	CACHE_SCRATCH,
	ctdb_lookup_path,
	fetch,
};
use cdtoc::{
	CtdbLookup,
	Toc,
};
use std::ops::Range;
use super::STRIDE;

/// Number of interleaved 16-bit words per stride.
const LANES: usize = STRIDE * 2;

/// A database pressing and its available reference data.
pub(super) struct Entry {
	/// CTDB's numeric entry ID.
	pub(super) id: u64,
	/// Number of submissions behind this checksum.
	pub(super) confidence: u16,
	/// CRC of the record's protected audio window.
	pub(super) disc_crc: u32,
	/// Track CRCs in the record's INDEX 01 coordinates.
	pub(super) checks: Vec<(Range<usize>, u32)>,
	/// Advertised syndromes for the first channel lane.
	pub(super) reference: Vec<u16>,
	/// HTTPS path of the full syndrome record, when available.
	pub(super) parity_url: Option<String>,
}

/// Reuse the normal verifier's cached lookup. A read-only caller can use this
/// path even when the rip cache has never been created.
pub(super) fn lookup(toc: &Toc, lengths: &[usize]) -> Option<Vec<Entry>> {
	let dst = ctdb_lookup_path(toc).ok()?;
	let raw = fetch(&dst, Some(toc.ctdb_lookup_url()))?;
	parse(&raw, toc, lengths)
}

/// Adapt cdtoc's compatible lookup records to the repair decoder's geometry.
fn parse(bytes: &[u8], toc: &Toc, lengths: &[usize]) -> Option<Vec<Entry>> {
	let xml = std::str::from_utf8(bytes).ok()?;
	let lookup = CtdbLookup::from_xml(xml, toc).ok()?;

	let mut entries = Vec::new();
	for record in lookup.entries() {
		let settings = record.parity_settings();
		if usize::from(settings.stride().get()) != STRIDE { continue; }

		let mut checks = Vec::with_capacity(lengths.len());
		// In this lookup format zero is also used for an unavailable track CRC.
		let mut start = 0;
		for (&len, &crc) in lengths.iter().zip(record.track_checksums()) {
			let end = start + len;
			if crc != 0 { checks.push((start..end, crc)); }
			start = end;
		}

		let parity_url = record.parity_data_url();
		entries.push(Entry {
			id: record.id(),
			confidence: record.confidence().get(),
			disc_crc: record.album_checksum(),
			checks,
			reference: settings.syndrome().to_vec(),
			parity_url,
		});
	}

	if entries.is_empty() { return None; }
	entries.sort_by_key(|entry| std::cmp::Reverse(entry.confidence));
	Some(entries)
}

/// Download the complete record, with rows of little-endian words per root.
pub(super) fn parity(entry: &Entry) -> Result<Vec<u16>, String> {
	let url = entry.parity_url.as_deref().ok_or("no supported parity URL")?;
	let bytes = cache_path(format!("{CACHE_SCRATCH}/ctdb-parity__{}.bin", entry.id))
		.ok()
		.and_then(|dst| fetch(&dst, Some(url)))
		.ok_or("unable to fetech parity.")?;

	let size = entry.reference.len() * LANES * 2;
	if bytes.len() != size {
		return Err("CTDB parity data size is incorrect.".into());
	}
	let stride_bytes = LANES * 2;
	if bytes.is_empty() || !bytes.len().is_multiple_of(stride_bytes) {
		return Err("CTDB parity does not contain complete syndrome rows".into());
	}
	let data = words(&bytes);
	if !entry.reference.iter().zip(data.iter().step_by(LANES)).all(|(a, b)| a == b) {
		return Err("CTDB parity disagrees with the lookup syndrome".into());
	}
	Ok(data)
}

/// Parse unsigned PCM field elements without depending on host byte order.
fn words(bytes: &[u8]) -> Vec<u16> {
	bytes.as_chunks::<2>().0.iter().copied().map(u16::from_le_bytes).collect()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn parses_live_v3_attributes_and_skips_incompatible_records() {
		let xml = br#"<ctdb xmlns="http://db.cuetools.net/ns/mmd-1.0#">
			<entry confidence="8" crc32="e0a03be5" hasparity="http://p.cuetools.net/94933"
				id="94933" npar="8" stride="5880" syndrome="U5VQSKpJeu1+GUqT6KNUrw=="
				toc="0:11413:25024:45713:55220" trackcrcs="c6858467 bd788446 d479ac2e 193890d4"/>
			<entry confidence="1" crc32="9b38421f" id="4700756" npar="16" stride="5880"
				syndrome="U5VQSKpJeu1+GUqT6KNUr89AVe1jHL/mwiMSSBI9X2w="
				toc="0:11413:25024:45713:55220" trackcrcs="c6858467 bd788446 dc379f45 193890d4"/>
			<entry confidence="12" crc32="00000001" id="2" npar="8" stride="5881"
				syndrome="U5VQSKpJeu1+GUqT6KNUrw==" toc="0:11413:25024:45713:55220"
				trackcrcs="c6858467 bd788446 d479ac2e 193890d4"/>
		</ctdb>"#;
		// The local disc has a pregap and a data session outside the audio image.
		let toc = Toc::from_parts(vec![250, 11_663, 25_274, 45_963], Some(66_870), 70_000).unwrap();
		let lengths: Vec<_> = toc.audio_tracks().map(|track| track.duration().samples() as usize).collect();
		let entries = parse(xml, &toc, &lengths).unwrap();
		assert_eq!(
			entries.len(),
			2,
			"unsupported stride is skipped and checksum-only entries are retained",
		);
		let entry = &entries[0];
		assert_eq!(entry.id, 94_933, "confidence order");
		assert_eq!(entry.disc_crc, 0xe0a0_3be5, "disc CRC");
		assert_eq!(entry.reference[0], 0x9553, "little-endian syndrome");
		assert_eq!(entry.checks[0], (0..11_413 * 588, 0xc685_8467), "track CRC range");
		assert_eq!(
			entry.parity_url.as_deref(),
			Some("http://p.cuetools.net/94933"),
			"verified S3 HTTPS host",
		);
		assert!(entries[1].parity_url.is_none(), "checksum-only entry");
	}
}
