# rm all proof files
find . -name "*.proof" -type f -delete
# delete all shared files
find . -name "*.shared" -type f -delete
# delete public input file and target folder
rm -rf public_input.json
rm -rf target
find . -name "verification_key" -type f -delete
# delete all bb proof files
find . -name "proof.bb*" -type f -delete
# delete all bb vk files
find . -name "vk.bb*" -type f -delete
# delete all proofs and vks in test_vectors files
find . -name "vk" -type f -delete
find . -name "proof" -type f -delete
