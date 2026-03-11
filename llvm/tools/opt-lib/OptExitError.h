
class OptExitError {
public:
  /// Create an error on exit helper.
  OptExitError(std::string Banner = "", int DefaultErrorExitCode = 1)
      : Banner(std::move(Banner)),
        GetExitCode([=](const llvm::Error &) { return DefaultErrorExitCode; }) {}

  /// Set the banner string for any errors caught by operator().
  void setBanner(std::string Banner) { this->Banner = std::move(Banner); }

  /// Set the exit-code mapper function.
  void setExitCodeMapper(std::function<int(const llvm::Error &)> GetExitCode) {
    this->GetExitCode = std::move(GetExitCode);
  }

  /// Check Err. If it's in a failure state log the error(s) and exit.
  void operator()(llvm::Error Err) const { checkError(std::move(Err)); }

  /// Check E. If it's in a success state then return the contained value. If
  /// it's in a failure state log the error(s) and exit.
  template <typename T> T operator()(llvm::Expected<T> &&E) const {
    checkError(E.takeError());
    return std::move(*E);
  }

  /// Check E. If it's in a success state then return the contained reference. If
  /// it's in a failure state log the error(s) and exit.
  template <typename T> T& operator()(llvm::Expected<T&> &&E) const {
    checkError(E.takeError());
    return *E;
  }

private:
  int checkError(llvm::Error Err) const {
    if (Err) {
      int ExitCode = GetExitCode(Err);
    //   logAllUnhandledErrors(std::move(Err), errs(), Banner);
      return ExitCode;
    }
  }

  std::string Banner;
  std::function<int(const llvm::Error &)> GetExitCode;
};