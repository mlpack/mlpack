//! Add a serialization operator.
template<typename Archive, typename eT>
void serialize(Archive& ar, arma::Cube<eT>& cube);
