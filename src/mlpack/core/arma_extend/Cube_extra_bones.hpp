//! Add a serialization operator.
template<typename eT>
template<typename Archive>
void serialize(Archive& ar, arma::Cube<eT>& cube);
