# The counts the prose asserts, recomputed from the files it is describing.
#
# A sentence like "best of its 30 runs, 10 IPF iterations by 3 seeds" is an
# arithmetic claim about results/transport.csv written as English, and English
# does not get recomputed when the experiment is rerun with different settings.
# Neither does "8000 points per side" or "Two of ninety DSB runs went non
# finite". Each of those is read here out of results/run-meta.json or counted
# out of results/transport.csv, turned back into the exact wording the document
# uses, and required to still be there.
#
# Numbers written as words are spelled out through the small table below rather
# than matched loosely, because a check that accepts either spelling accepts a
# document that says both.

require "csv"
require "json"

root = ARGV[0] || "."
# Ruby 2.6 opens files as US-ASCII, and both documents contain non ASCII, so the
# encoding has to be given or reading raises before anything can be matched.
# Line wrapping is not a change of claim, so runs of whitespace collapse to one
# space before anything is matched.
def read(path)
  File.read(path, encoding: "UTF-8").gsub(/[*`]/, "")
end

def flatten(text)
  text.gsub(/\s+/, " ")
end

readme_raw  = read(File.join(root, "README.md"))
methods_raw = read(File.join(root, "notes", "METHODS.md"))
readme  = flatten(readme_raw)
methods = flatten(methods_raw)
meta    = JSON.parse(read(File.join(root, "results", "run-meta.json")))
rows    = CSV.read(File.join(root, "results", "transport.csv"), headers: true)

WORDS = { 1 => "one", 2 => "two", 3 => "three", 4 => "four", 8 => "eight",
          9 => "nine", 10 => "ten", 90 => "ninety" }

def word(n)
  WORDS.fetch(n) { raise "no spelling for #{n}" }
end

bad = 0
def check(doc, name, phrase, bad)
  if doc.downcase.include?(phrase.downcase)
    puts format("  %-40s %s", name, phrase.inspect)
    bad
  else
    puts format("  %-40s %s NOT FOUND", name, phrase.inspect)
    bad + 1
  end
end

dsb = rows.select { |r| r["method"] == "dsb" }
nonfinite = dsb.select { |r| !(Float(r["w2"]) rescue Float::NAN).finite? }
per_pair = dsb.group_by { |r| r["pair"] }.map { |_, v| v.size }.uniq
raise "dsb rows per pair are not all equal: #{per_pair}" if per_pair.size != 1

bad = check(readme, "points per side", "#{meta['n']} points per side", bad)
bad = check(readme, "seeds", "median of #{meta['seeds'].size} seeds", bad)
bad = check(readme, "dsb runs per pair",
            "best of its #{per_pair[0]} runs, #{meta['dsb_ipf']} IPF iterations " \
            "by #{meta['seeds'].size} seeds", bad)
bad = check(readme, "IPF iterations in the cost table",
            "#{meta['dsb_ipf']} IPF iterations, each simulating and training", bad)

stages = nonfinite.map { |r| r["stage"].to_i }.sort
pairs  = nonfinite.map { |r| r["pair"] }.uniq
raise "the non finite runs are not on one pair: #{pairs}" if pairs.size != 1
bad = check(methods, "non finite dsb runs",
            "#{word(nonfinite.size).capitalize} of #{word(dsb.size)} DSB runs went non finite", bad)
bad = check(methods, "where they went non finite",
            "both on #{pairs[0].sub('->', ' to ')} at iterations " \
            "#{stages[0]} and #{stages[1]}", bad)

# The METHODS table has one column per IPF iteration. If the experiment is rerun
# with a different dsb_ipf the header stops describing it.
header = methods_raw.lines.find { |l| l.start_with?("| pair | ipf 0 |") }
raise "no IPF table header in notes/METHODS.md" if header.nil?
cols = header.split("|").map(&:strip).reject(&:empty?).size - 1
if cols == meta["dsb_ipf"]
  puts format("  %-40s %d columns", "IPF table width", cols)
else
  puts format("  %-40s %d columns, run-meta says %d iterations",
              "IPF table width", cols, meta["dsb_ipf"])
  bad += 1
end

if bad > 0
  puts "  #{bad} counts in the prose no longer match results/"
  exit 1
end
puts "  every count asserted in words matches results/run-meta.json and transport.csv"
exit 0
