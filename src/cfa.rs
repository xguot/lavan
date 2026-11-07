//! A bounded subset of lavaan's CFA front end.
//!
//! Supports measurement relations (`=~`), observed residual / latent covariances
//! (`~~`), and intercepts (`~ 1`), with numeric or label modifiers. Other syntax
//! is rejected explicitly. Full lavaan parsing and defaults remain unported.

use crate::{
    Error, Matrix, Result,
    model::{Fit, Lisrel, Member, Model, Parameter, Slot},
    optim,
    sample::SampleStats,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, Default)]
pub struct CfaOptions {
    pub meanstructure: bool,
    pub std_lv: bool,
    pub orthogonal: bool,
    pub optimizer: optim::Options,
}

#[derive(Debug, Clone)]
pub struct CfaFit {
    pub fit: Fit,
    pub model: Model,
    pub observed_names: Vec<String>,
    pub latent_names: Vec<String>,
}

#[derive(Debug, Clone)]
struct Term {
    lhs: String,
    op: &'static str,
    rhs: String,
    fixed: Option<f64>,
    free: bool,
    label: Option<String>,
}

fn identifier(s: &str) -> bool {
    let mut chars = s.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_' || c == '.')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

fn split_terms(s: &str) -> Vec<&str> {
    let mut start = 0;
    let mut terms = Vec::new();
    let bytes = s.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] == b'+'
            && !(i > 0
                && matches!(bytes[i - 1], b'e' | b'E')
                && i + 1 < bytes.len()
                && bytes[i + 1].is_ascii_digit())
        {
            terms.push(&s[start..i]);
            start = i + 1;
        }
    }
    terms.push(&s[start..]);
    terms
}

fn parse(syntax: &str) -> Result<Vec<Term>> {
    let mut statements: Vec<String> = Vec::new();
    for raw in syntax
        .lines()
        .flat_map(|line| line.split('#').next().unwrap_or("").split(';'))
    {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line.contains('~') {
            statements.push(line.to_owned());
        } else if let Some(last) = statements.last_mut() {
            last.push(' ');
            last.push_str(line);
        } else {
            return Err(Error::Unsupported(format!("CFA syntax: {line}")));
        }
    }
    let mut terms = Vec::new();
    for statement in statements {
        let (op, position) = if let Some(i) = statement.find("=~") {
            ("=~", i)
        } else if let Some(i) = statement.find("~~") {
            ("~~", i)
        } else if let Some(i) = statement.find('~') {
            ("~", i)
        } else {
            return Err(Error::Unsupported(format!("CFA syntax: {statement}")));
        };
        let lhs = statement[..position].trim();
        if !identifier(lhs) {
            return Err(Error::Unsupported(format!("CFA left hand side: {lhs}")));
        }
        for raw in split_terms(&statement[position + op.len()..]) {
            let pieces: Vec<&str> = raw.trim().split('*').map(str::trim).collect();
            if pieces.len() > 2 || pieces.is_empty() {
                return Err(Error::Unsupported(format!("CFA modifier: {raw}")));
            }
            let rhs = pieces[pieces.len() - 1];
            if !(identifier(rhs) || op == "~" && rhs == "1") {
                return Err(Error::Unsupported(format!("CFA right hand side: {rhs}")));
            }
            if op == "~" && rhs != "1" {
                return Err(Error::Unsupported(
                    "CFA regressions; use the numeric model API".into(),
                ));
            }
            let (mut fixed, mut label, mut free) = (None, None, false);
            if pieces.len() == 2 {
                if pieces[0] == "NA" {
                    free = true;
                } else if let Ok(value) = pieces[0].parse::<f64>() {
                    if !value.is_finite() {
                        return Err(Error::InvalidInput("non-finite CFA modifier".into()));
                    }
                    fixed = Some(value);
                } else if identifier(pieces[0]) {
                    label = Some(pieces[0].to_owned());
                } else {
                    return Err(Error::Unsupported(format!("CFA modifier: {}", pieces[0])));
                }
            }
            terms.push(Term {
                lhs: lhs.to_owned(),
                op,
                rhs: rhs.to_owned(),
                fixed,
                label,
                free,
            });
        }
    }
    if terms.is_empty() || !terms.iter().any(|t| t.op == "=~") {
        return Err(Error::InvalidInput(
            "CFA requires a measurement model".into(),
        ));
    }
    Ok(terms)
}

fn push_unique(names: &mut Vec<String>, name: &str) {
    if !names.iter().any(|s| s == name) {
        names.push(name.to_owned());
    }
}

#[derive(Clone)]
struct Entry {
    slot: Slot,
    name: String,
    fixed: Option<f64>,
    initial: f64,
    label: Option<String>,
}

/// Fit the supported CFA syntax on complete continuous data using normal ML.
/// Data columns are matched by name and the ML covariance uses divisor N.
pub fn cfa(syntax: &str, data: &Matrix, columns: &[&str], options: CfaOptions) -> Result<CfaFit> {
    if columns.len() != data.ncols() {
        return Err(Error::Dimension("column name count differs".into()));
    }
    let unique: BTreeSet<&str> = columns.iter().copied().collect();
    if unique.len() != columns.len() {
        return Err(Error::InvalidInput("duplicate data column names".into()));
    }
    let terms = parse(syntax)?;
    let mut latent = Vec::new();
    let mut observed = Vec::new();
    for term in &terms {
        if term.op == "=~" {
            push_unique(&mut latent, &term.lhs);
        }
    }
    for term in &terms {
        if term.op == "=~" {
            if latent.contains(&term.rhs) {
                return Err(Error::Unsupported(
                    "higher-order CFA syntax; use the numeric model API".into(),
                ));
            }
            push_unique(&mut observed, &term.rhs);
        }
    }
    for term in &terms {
        for name in [&term.lhs, &term.rhs] {
            if name == "1" {
                continue;
            }
            if !latent.contains(name) && !observed.contains(name) {
                return Err(Error::Unsupported(format!(
                    "variable {name} has no measurement relation"
                )));
            }
        }
    }
    let indices: Vec<usize> = observed
        .iter()
        .map(|name| {
            columns
                .iter()
                .position(|s| *s == name)
                .ok_or_else(|| Error::InvalidInput(format!("data lacks variable {name}")))
        })
        .collect::<Result<_>>()?;
    let selected = Matrix::from_fn(data.nrows(), indices.len(), |i, j| data[(i, indices[j])]);
    let stats = SampleStats::from_data(&selected, None)?;
    let (p, q) = (observed.len(), latent.len());
    let meanstructure = options.meanstructure || terms.iter().any(|t| t.op == "~");
    let mut entries: BTreeMap<Slot, Entry> = BTreeMap::new();
    let mut explicitly_set = BTreeSet::new();
    let mut first_loading = BTreeSet::new();
    for term in terms.iter().filter(|t| t.op == "=~") {
        let row = observed.iter().position(|n| n == &term.rhs).unwrap();
        let col = latent.iter().position(|n| n == &term.lhs).unwrap();
        let slot = Slot {
            member: Member::Lambda,
            row,
            col,
        };
        if !explicitly_set.insert(slot) {
            return Err(Error::InvalidInput("duplicate measurement relation".into()));
        }
        let first = first_loading.insert(col);
        let fixed = term
            .fixed
            .or_else(|| (first && !options.std_lv && !term.free).then_some(1.0));
        entries.insert(
            slot,
            Entry {
                slot,
                name: format!("{}=~{}", term.lhs, term.rhs),
                fixed,
                initial: 0.7,
                label: term.label.clone(),
            },
        );
    }
    for (i, observed_name) in observed.iter().enumerate() {
        let slot = Slot {
            member: Member::Theta,
            row: i,
            col: i,
        };
        entries.insert(
            slot,
            Entry {
                slot,
                name: format!("{observed_name}~~{observed_name}"),
                fixed: None,
                initial: stats.covariance()[(i, i)] * 0.5,
                label: None,
            },
        );
        if meanstructure {
            let slot = Slot {
                member: Member::Nu,
                row: i,
                col: 0,
            };
            entries.insert(
                slot,
                Entry {
                    slot,
                    name: format!("{observed_name}~1"),
                    fixed: None,
                    initial: stats.mean()[i],
                    label: None,
                },
            );
        }
    }
    for j in 0..q {
        for i in j..q {
            let slot = Slot {
                member: Member::Psi,
                row: i,
                col: j,
            };
            let fixed = if i == j {
                options.std_lv.then_some(1.0)
            } else {
                options.orthogonal.then_some(0.0)
            };
            entries.insert(
                slot,
                Entry {
                    slot,
                    name: format!("{}~~{}", latent[i], latent[j]),
                    fixed,
                    initial: if i == j { 1.0 } else { 0.0 },
                    label: None,
                },
            );
        }
    }
    for term in terms.iter().filter(|t| t.op != "=~") {
        let slot = if term.op == "~" {
            if let Some(row) = observed.iter().position(|n| n == &term.lhs) {
                Slot {
                    member: Member::Nu,
                    row,
                    col: 0,
                }
            } else {
                Slot {
                    member: Member::Alpha,
                    row: latent.iter().position(|n| n == &term.lhs).unwrap(),
                    col: 0,
                }
            }
        } else {
            let ov = (
                observed.iter().position(|n| n == &term.lhs),
                observed.iter().position(|n| n == &term.rhs),
            );
            let lv = (
                latent.iter().position(|n| n == &term.lhs),
                latent.iter().position(|n| n == &term.rhs),
            );
            if let (Some(i), Some(j)) = ov {
                Slot {
                    member: Member::Theta,
                    row: i.max(j),
                    col: i.min(j),
                }
            } else if let (Some(i), Some(j)) = lv {
                Slot {
                    member: Member::Psi,
                    row: i.max(j),
                    col: i.min(j),
                }
            } else {
                return Err(Error::Unsupported(
                    "observed-latent covariance syntax; use the numeric model API".into(),
                ));
            }
        };
        if !explicitly_set.insert(slot) {
            return Err(Error::InvalidInput(
                "duplicate CFA parameter statement".into(),
            ));
        }
        let initial = entries.get(&slot).map_or(0.0, |e| e.initial);
        entries.insert(
            slot,
            Entry {
                slot,
                name: format!("{}{}{}", term.lhs, term.op, term.rhs),
                fixed: term.fixed,
                initial,
                label: term.label.clone(),
            },
        );
    }
    let mut base = Lisrel::new(
        Matrix::zeros(p, q),
        Matrix::zeros(p, p),
        Matrix::zeros(q, q),
    )?;
    let mut parameters: Vec<Parameter> = Vec::new();
    let mut labels = BTreeMap::<String, usize>::new();
    let mut fixed_labels = BTreeMap::<String, f64>::new();
    for entry in entries.values() {
        if let (Some(label), Some(value)) = (&entry.label, entry.fixed)
            && let Some(previous) = fixed_labels.insert(label.clone(), value)
            && previous != value
        {
            return Err(Error::InvalidInput("inconsistent fixed labels".into()));
        }
    }
    for entry in entries.into_values() {
        let fixed = entry.fixed.or_else(|| {
            entry
                .label
                .as_ref()
                .and_then(|label| fixed_labels.get(label).copied())
        });
        let value = fixed.unwrap_or(entry.initial);
        match entry.slot.member {
            Member::Lambda => base.lambda[(entry.slot.row, entry.slot.col)] = value,
            Member::Theta => {
                base.theta[(entry.slot.row, entry.slot.col)] = value;
                base.theta[(entry.slot.col, entry.slot.row)] = value;
            }
            Member::Psi => {
                base.psi[(entry.slot.row, entry.slot.col)] = value;
                base.psi[(entry.slot.col, entry.slot.row)] = value;
            }
            Member::Nu => base.nu[entry.slot.row] = value,
            Member::Alpha => base.alpha[entry.slot.row] = value,
            Member::Beta => unreachable!(),
        }
        if fixed.is_some() {
            continue;
        }
        if let Some(id) = entry.label.as_ref().and_then(|s| labels.get(s).copied()) {
            parameters[id].slots.push(entry.slot);
        } else {
            if let Some(label) = &entry.label {
                labels.insert(label.clone(), parameters.len());
            }
            parameters.push(Parameter {
                name: entry.label.unwrap_or(entry.name),
                initial: entry.initial,
                slots: vec![entry.slot],
            });
        }
    }
    let model = Model::new(base, parameters, meanstructure)?;
    let fit = model.fit(stats, options.optimizer)?;
    Ok(CfaFit {
        fit,
        model,
        observed_names: observed,
        latent_names: latent,
    })
}
