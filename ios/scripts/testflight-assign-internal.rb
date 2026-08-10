#!/usr/bin/env ruby
# frozen_string_literal: true

require "base64"
require "json"
require "net/http"
require "openssl"
require "time"
require "uri"

API_ORIGIN = "https://api.appstoreconnect.apple.com"
DEFAULT_BUNDLE_ID = "com.tiltastech.zimfo"
DEFAULT_GROUP = "InternalTesters"

def abort_with(message)
  warn "testflight: #{message}"
  exit 1
end

def base64url(bytes)
  Base64.urlsafe_encode64(bytes, padding: false)
end

def fixed_width(integer, width)
  bytes = integer.to_s(2)
  abort_with("invalid ES256 signature component") if bytes.bytesize > width

  ("\0" * (width - bytes.bytesize)) + bytes
end

def app_store_token(key_id:, issuer_id:, key_path:)
  now = Time.now.to_i
  header = base64url(JSON.generate(alg: "ES256", kid: key_id, typ: "JWT"))
  payload = base64url(JSON.generate(iss: issuer_id, iat: now, exp: now + 1_200, aud: "appstoreconnect-v1"))
  signing_input = "#{header}.#{payload}"
  key = OpenSSL::PKey::EC.new(File.read(key_path))
  der_signature = key.sign(OpenSSL::Digest::SHA256.new, signing_input)
  sequence = OpenSSL::ASN1.decode(der_signature)
  raw_signature = fixed_width(sequence.value.fetch(0).value, 32) +
                  fixed_width(sequence.value.fetch(1).value, 32)
  "#{signing_input}.#{base64url(raw_signature)}"
end

class AppStoreConnect
  def initialize(token)
    @token = token
  end

  def get(path, query = {})
    request(Net::HTTP::Get, path, query: query)
  end

  def post(path, body)
    request(Net::HTTP::Post, path, body: body)
  end

  private

  def request(request_class, path, query: {}, body: nil)
    uri = URI.join(API_ORIGIN, path)
    uri.query = URI.encode_www_form(query) unless query.empty?
    request = request_class.new(uri)
    request["Authorization"] = "Bearer #{@token}"
    request["Content-Type"] = "application/json"
    request.body = JSON.generate(body) if body

    response = Net::HTTP.start(uri.host, uri.port, use_ssl: true) { |http| http.request(request) }
    parsed = response.body.nil? || response.body.empty? ? {} : JSON.parse(response.body)
    return parsed if response.is_a?(Net::HTTPSuccess)

    details = Array(parsed["errors"]).map { |error| error["detail"] || error["title"] }.compact.join("; ")
    abort_with("App Store Connect #{request.method} #{uri.path} failed (#{response.code}): #{details}")
  rescue JSON::ParserError
    abort_with("App Store Connect #{request.method} #{uri.path} returned invalid JSON (#{response.code})")
  end
end

def one_matching(records, description, &block)
  matches = records.select(&block)
  abort_with("could not find #{description}") if matches.empty?
  abort_with("found multiple matches for #{description}") if matches.length > 1

  matches.first
end

build_number = ARGV.fetch(0) { abort_with("usage: #{$PROGRAM_NAME} BUILD_NUMBER [GROUP_NAME]") }
group_name = ARGV.fetch(1, ENV.fetch("MCPZIM_INTERNAL_GROUP", DEFAULT_GROUP))
bundle_id = ENV.fetch("MCPZIM_BUNDLE_ID", DEFAULT_BUNDLE_ID)
key_id = ENV["ASC_KEY_ID"] || abort_with("ASC_KEY_ID is not set")
issuer_id = ENV["ASC_ISSUER_ID"] || abort_with("ASC_ISSUER_ID is not set")
key_path = ENV["ASC_KEY_PATH"] || abort_with("ASC_KEY_PATH is not set")
abort_with("App Store Connect key not found at #{key_path}") unless File.file?(key_path)

client = AppStoreConnect.new(app_store_token(key_id: key_id, issuer_id: issuer_id, key_path: key_path))

apps = client.get("/v1/apps", "filter[bundleId]" => bundle_id, "limit" => 10).fetch("data", [])
app = one_matching(apps, "app #{bundle_id}") { |item| item.dig("attributes", "bundleId") == bundle_id }
app_id = app.fetch("id")

groups = client.get("/v1/betaGroups", "filter[app]" => app_id, "limit" => 200).fetch("data", [])
internal_groups = groups.select { |item| item.dig("attributes", "isInternalGroup") == true }
group_matches = internal_groups.select do |item|
  item.dig("attributes", "name") == group_name && item.dig("attributes", "isInternalGroup") == true
end
if group_matches.empty?
  names = internal_groups.map { |item| item.dig("attributes", "name") }.compact.sort
  abort_with("could not find internal group #{group_name.inspect}; available internal groups: #{names.join(", ")}")
end
abort_with("found multiple matches for internal group #{group_name.inspect}") if group_matches.length > 1
group = group_matches.first
group_id = group.fetch("id")

deadline = Time.now + Integer(ENV.fetch("MCPZIM_PROCESSING_TIMEOUT", "900"))
build = nil
loop do
  builds = client.get(
    "/v1/builds",
    "filter[app]" => app_id,
    "filter[version]" => build_number,
    "limit" => 10
  ).fetch("data", [])
  build = one_matching(builds, "build #{build_number}") do |item|
    item.dig("attributes", "version") == build_number
  end
  state = build.dig("attributes", "processingState")
  break if state == "VALID"
  abort_with("build #{build_number} processing failed (#{state})") if state == "FAILED" || state == "INVALID"
  abort_with("timed out waiting for build #{build_number} processing (#{state})") if Time.now >= deadline

  puts "testflight: build #{build_number} processing state = #{state}; waiting 30s"
  sleep 30
end

build_id = build.fetch("id")
beta_detail = nil
loop do
  beta_detail = client.get("/v1/builds/#{build_id}/buildBetaDetail")["data"]
  internal_state = beta_detail&.dig("attributes", "internalBuildState")
  break if %w[READY_FOR_BETA_TESTING IN_BETA_TESTING].include?(internal_state)
  if %w[EXPIRED INVALID].include?(internal_state)
    abort_with("build #{build_number} cannot enter internal testing (#{internal_state})")
  end
  if Time.now >= deadline
    attributes = build.fetch("attributes", {}).slice("version", "uploadedDate", "expirationDate", "expired", "processingState", "usesNonExemptEncryption")
    abort_with("timed out waiting for TestFlight detail for build #{build_number}; internal state=#{internal_state.inspect}; build attributes=#{JSON.generate(attributes)}")
  end

  puts "testflight: build #{build_number} internal state = #{internal_state || "not available"}; waiting 30s"
  sleep 30
end
internal_state = beta_detail.dig("attributes", "internalBuildState")
puts "testflight: selected build #{build_number} · id=#{build_id} · internal state=#{internal_state}"

relationship_path = "/v1/betaGroups/#{group_id}/relationships/builds"
related = client.get(relationship_path, "limit" => 200).fetch("data", [])
unless related.any? { |item| item["id"] == build_id }
  client.post(relationship_path, data: [{ type: "builds", id: build_id }])
end

verified = client.get(relationship_path, "limit" => 200).fetch("data", [])
abort_with("build #{build_number} was not present in #{group_name.inspect} after assignment") unless verified.any? { |item| item["id"] == build_id }

puts "testflight: assigned Zimfo build #{build_number} to internal group #{group_name.inspect}"
