#include <mfem.hpp>
#include <print>
#include <memory>
#include <cmath>
#include <string>
#include <functional>
#include <expected>
#include <map>

#include <CLI/CLI.hpp>

constexpr double G = 1.0;
constexpr double MASS = 1.0;
constexpr double RADIUS = 1.0;
constexpr double CENTRAL_DENSITY = 0.9550104783;
constexpr double mP = 1.0;
constexpr double kB = 0.28;

constexpr char HOST[10] = "localhost";
constexpr int PORT = 19916;


/*****************************
*
*  Types
*
* ***************************/

class MassContinuitySolver {
  private:
    mfem::FiniteElementSpace &m_fes;
    std::unique_ptr<mfem::BilinearForm> m_laplacian;
  public:
    explicit MassContinuitySolver(mfem::FiniteElementSpace& fes);
    void Solve(mfem::Coefficient& rho, mfem::GridFunction& phi_gf, const mfem::Array<int>& ess_tdof_list) const;
};

struct FEM {
  std::unique_ptr<mfem::Mesh> mesh_ptr;

  std::unique_ptr<mfem::FiniteElementCollection> H1_fec;
  std::unique_ptr<mfem::FiniteElementCollection> L2_fec;
  
  std::unique_ptr<mfem::FiniteElementSpace> H1_fes;
  std::unique_ptr<mfem::FiniteElementSpace> L2_fes;

  std::unique_ptr<mfem::FiniteElementSpace> Vec_H1_fes;

  std::unique_ptr<mfem::GridFunction> rho_gf;
  std::unique_ptr<mfem::GridFunction> phi_gf;
  std::unique_ptr<mfem::GridFunction> H_gf;
  std::unique_ptr<mfem::GridFunction> P_gf;

  [[nodiscard]] bool okay() const;
};

struct FixedPoint {
  std::unique_ptr<mfem::GridFunction> rho;
  std::unique_ptr<mfem::GridFunction> h;
  std::unique_ptr<mfem::GridFunction> phi;

  [[nodiscard]] FixedPoint clone() const;
};

enum class FixedPointErrors : uint8_t {
  UNBOUNDED,
  MAX_ITERS
};

static const std::map<FixedPointErrors, const char*> FixedPointErrorMessages = {
  {FixedPointErrors::UNBOUNDED, "The system appears to be unbounded. Try reducing the rotation rate or increasing the polytropic index."},
  {FixedPointErrors::MAX_ITERS, "Maximum number of iterations reached without convergence. Try increasing max iterations or relaxing the convergence criteria."}
};

enum class Verbosity : uint8_t {
  SILENT,
  PER_DEFORMATION,
  PER_ITERATION,
  FULL,
  VERBOSE
};

static const std::map<Verbosity, std::pair<const char*,const char*>> VerbosityNames = {
  {Verbosity::SILENT, {"S", "SILENT"}},
  {Verbosity::PER_DEFORMATION, {"D", "PER_DEFORMATION"}},
  {Verbosity::PER_ITERATION, {"I", "PER_ITERATION"}},
  {Verbosity::FULL, {"F", "FULL"}},
  {Verbosity::VERBOSE, {"V", "VERBOSE"}}
};

struct Point {
  double x;
  double y;
  double z;
  double r;
  size_t IR_ID;
  size_t BE_ID;
};

struct Envelope {
  std::vector<Point> points;
};


struct Args {
  bool visualize = false;
  bool rotation = true;
  Verbosity verbosity = Verbosity::PER_DEFORMATION;
  double index = 1;
  double alpha = 0.1;
  std::string mesh = "stroid.mesh";
  size_t max_iters = 500;
  double tol = 1e-8;
  double omega = 1.0;
  double relax_rate = 0.1;
  bool allow_deformation = true;
  size_t max_deformation_iters = 50;
  double deformation_rtol = 1e-4;
  double deformation_atol = 1e-6;
  int env_ref_levels = 2;
};


/*****************************
*
*  Functions
*
* ***************************/
FEM setup_fem(const std::string& filename, bool verbose=true);

void ViewMesh(const std::string& host, int port, const mfem::Mesh& mesh, const mfem::GridFunction& gf, const std::string& title);

double initial_density(const mfem::Vector& x);

double get_current_mass(const FEM& fem, const mfem::GridFunction& gf);

double centrifugal_potential(const mfem::Vector& x, double omega);

void project_scalar_function(mfem::GridFunction& gf, const std::function<double(const mfem::Vector& x)> &g);

std::unique_ptr<mfem::GridFunction> get_potential(const FEM& fem, const mfem::GridFunction& rho, const Args& args);

std::unique_ptr<mfem::GridFunction> get_enthalpy(const FEM& fem, const mfem::GridFunction& phi);

double get_polar_value(const mfem::GridFunction& gf, mfem::Mesh& mesh, double radius);

double gamma(double n);

double rho_from_enthalpy_barotropic(double h, double n);

double mix_density(double rho_0, double rho_1, double alpha);

std::unique_ptr<mfem::GridFunction> update_density(const FEM& fem, const FixedPoint& fp_old, const FixedPoint& fp_new, double n, double alpha);

std::unique_ptr<mfem::GridFunction> conserve_mass(const FEM& fem, const mfem::GridFunction& gf, double target_mass);

FixedPoint init_fp(const FEM& fem, const Args& args);

FixedPoint step(const FEM& fem, const FixedPoint& fp, const Args& args);

void VisualizeFP(const FEM& fem, const FixedPoint& fp, const std::string& prefix);

double L2RelativeResidual(const FixedPoint& fp_old, const FixedPoint& fp_new);

std::expected<FixedPoint, FixedPointErrors> iterate_for_constant_shape(const FEM &fem, const Args &args);

double clip(double h, double relax);

void radial(const mfem::Vector& x, mfem::Vector &r_hat);

bool is_system_bound(const FEM& fem, const FixedPoint& fp, const Args& args);

std::unique_ptr<mfem::GridFunction> get_nodal_displacement(const FEM& fem, const FixedPoint &fp, const Args& args);

void deform_mesh(const FEM& fem, const mfem::GridFunction& displacement, const Args& args);

Args setup_cli(int argc, char** argv);

void print_options(const Args& args);

Envelope extract_envelope(const FEM& fem, const Args& args);

std::map<std::string, Verbosity> make_verbosity_map();


/*****************************
*
*  Entry Point
*
* ***************************/

int main(const int argc, char** argv) {
  const Args args = setup_cli(argc, argv);
  print_options(args);

  FEM fem = setup_fem(args.mesh, args.verbosity == Verbosity::VERBOSE);

  double last_displacement_norm = std::numeric_limits<double>::infinity();
  for (int i = 0; i < args.max_deformation_iters; ++i) {
    std::print("Deformation Step {:3}{}", i, args.verbosity > Verbosity::PER_DEFORMATION ? "\n" : " -- ");
    std::cout << std::flush;
    const auto solution = iterate_for_constant_shape(fem, args);
    if (!solution) {
      std::cerr << "Error: " << FixedPointErrorMessages.at(solution.error()) << std::endl;
      exit(1);
    }
    const auto boundary_displacement = get_nodal_displacement(fem, solution.value(), args);

    double displacement_norm = std::numeric_limits<double>::infinity();
    if (i > 0) {
      displacement_norm = boundary_displacement->Norml2();
    }

    double rel_displacement = std::numeric_limits<double>::infinity();
    if (i > 3) {
      rel_displacement = (last_displacement_norm > 1e-18) ? (displacement_norm / last_displacement_norm) : displacement_norm;
    }

    if (args.verbosity >= Verbosity::PER_DEFORMATION) {
      std::println("||Da|| = {:5.3E}, ||Dr|| = {:5.3E}", (i > 0) ? displacement_norm : 0.0,  (i > 3) ? rel_displacement : 0.0);
    }

    if (displacement_norm <= args.deformation_atol || rel_displacement <= args.deformation_rtol) {
      if (args.verbosity >= Verbosity::PER_DEFORMATION) {
        std::println("Deformation convergence reached in {} steps!", i);
      }
      break;
    }

    last_displacement_norm = displacement_norm;

    deform_mesh(fem, *boundary_displacement, args);
    fem.rho_gf = std::make_unique<mfem::GridFunction>(*solution.value().rho);


  }
  auto solution = iterate_for_constant_shape(fem, args);
  if (!solution) {
    std::cerr << "Error: " << FixedPointErrorMessages.at(solution.error()) << std::endl;
    exit(1);
  }

  ViewMesh(HOST, PORT, *fem.mesh_ptr, *solution.value().phi, "Final Potential");
  ViewMesh(HOST, PORT, *fem.mesh_ptr, *solution.value().h, "Final Enthalpy");
  ViewMesh(HOST, PORT, *fem.mesh_ptr, *solution.value().rho, "Final Density");
}


/*****************************
*
*  Implementations
*
* ***************************/

FEM setup_fem(const std::string& filename, bool verbose) {
  FEM fem_setup;
  fem_setup.mesh_ptr = std::make_unique<mfem::Mesh>(filename, 0, 0);
  fem_setup.mesh_ptr->EnsureNodes();
  
  int dim = fem_setup.mesh_ptr->Dimension();
  
  fem_setup.H1_fec   = std::make_unique<mfem::H1_FECollection>(2, dim);
  fem_setup.L2_fec   = std::make_unique<mfem::L2_FECollection>(2, dim);

  fem_setup.H1_fes   = std::make_unique<mfem::FiniteElementSpace>(fem_setup.mesh_ptr.get(), fem_setup.H1_fec.get());
  fem_setup.L2_fes   = std::make_unique<mfem::FiniteElementSpace>(fem_setup.mesh_ptr.get(), fem_setup.L2_fec.get());

  fem_setup.Vec_H1_fes = std::make_unique<mfem::FiniteElementSpace>(fem_setup.mesh_ptr.get(), fem_setup.H1_fec.get(), dim, mfem::Ordering::byNODES);

  fem_setup.rho_gf   = std::make_unique<mfem::GridFunction>(fem_setup.H1_fes.get());
  fem_setup.phi_gf   = std::make_unique<mfem::GridFunction>(fem_setup.H1_fes.get());
  fem_setup.H_gf     = std::make_unique<mfem::GridFunction>(fem_setup.H1_fes.get());
  fem_setup.P_gf     = std::make_unique<mfem::GridFunction>(fem_setup.H1_fes.get());

  project_scalar_function(*fem_setup.rho_gf, initial_density);
  fem_setup.rho_gf = conserve_mass(fem_setup, *fem_setup.rho_gf, MASS);

  if (verbose) {
    std::println("Setup {}", fem_setup.okay() ? "OK" : "FAIL");
  }
  if (!fem_setup.okay()) {
    exit(1);
  }

  return fem_setup;
}

bool FEM::okay() const {
  const bool has_mesh = mesh_ptr != nullptr;

  const bool has_fec = (H1_fec != nullptr) && (L2_fec != nullptr);
  const bool has_fes = (H1_fes != nullptr) && (L2_fes != nullptr);

  const bool has_vec_fes = Vec_H1_fes != nullptr;

  const bool has_gf = (rho_gf != nullptr) && (phi_gf != nullptr) && (H_gf != nullptr) && (P_gf != nullptr);

  return has_mesh && has_fec && has_fes && has_gf && has_vec_fes;

}

FixedPoint FixedPoint::clone() const {
  FixedPoint fp;
  if (rho) fp.rho = std::make_unique<mfem::GridFunction>(*rho);
  if (h)   fp.h   = std::make_unique<mfem::GridFunction>(*h);
  if (phi) fp.phi = std::make_unique<mfem::GridFunction>(*phi);

  return fp;
}

double initial_density(const mfem::Vector& x) {
  const double r = x.Norml2();
  const double rho = MASS * CENTRAL_DENSITY * (1.0 - r / RADIUS);
  return rho;
}

double get_current_mass(const FEM& fem, const mfem::GridFunction& gf) {
  mfem::ConstantCoefficient one(1.0);
  mfem::LinearForm mass_lf(fem.H1_fes.get());
  mfem::GridFunctionCoefficient rho_coeff(&gf);

  mass_lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(rho_coeff));
  mass_lf.Assemble();

  const double current_mass = mass_lf.Sum();

  return current_mass;
}

double centrifugal_potential(const mfem::Vector& x, double omega) {
  const double s2 = std::pow(x(0), 2) + std::pow(x(1), 2);
  return -0.5 * s2 * std::pow(omega, 2);
}


MassContinuitySolver::MassContinuitySolver(mfem::FiniteElementSpace& fes): m_fes(fes) {
  m_laplacian = std::make_unique<mfem::BilinearForm>(&m_fes);
  m_laplacian->AddDomainIntegrator(new mfem::DiffusionIntegrator());
  m_laplacian->Assemble();
  m_laplacian->Finalize();
}

void MassContinuitySolver::Solve(mfem::Coefficient& rho, mfem::GridFunction& phi_gf, const mfem::Array<int>& ess_tdof_list) const {
  mfem::ConstantCoefficient fourPiG (-4.0 * M_PI * G);
  mfem::ProductCoefficient rhsCoeff(fourPiG, rho);
  mfem::LinearForm b(&m_fes);

  b.AddDomainIntegrator(new mfem::DomainLFIntegrator(rhsCoeff));
  b.Assemble();

  mfem::OperatorPtr A;
  mfem::Vector B, X;

  m_laplacian -> FormLinearSystem(ess_tdof_list, phi_gf, b, A, X, B);

  mfem::GSSmoother prec;
  mfem::CGSolver cg;
  cg.SetRelTol(1e-12);
  cg.SetMaxIter(2000);
  cg.SetPrintLevel(0);
  cg.SetPreconditioner(prec);
  cg.SetOperator(*A);

  cg.Mult(B, X);
  m_laplacian->RecoverFEMSolution(X, b, phi_gf);
}

void ViewMesh(const std::string& host, int port, const mfem::Mesh& mesh, const mfem::GridFunction& gf, const std::string& title) {
  mfem::socketstream sol_sock(host.c_str(), port);
  if (!sol_sock.is_open()) {
    std::cerr << "Unable to connect to GLVis Server running at " << host << ":" << port << std::endl; 
    return;
  }

  sol_sock.precision(8);
  sol_sock << "solution\n" << mesh << gf;
  sol_sock << "window_title '" << title << "'\n";
  sol_sock << "keys 'iIzzMaagpmtppcizzz'";
  sol_sock << std::flush;
}

std::unique_ptr<mfem::GridFunction> get_potential(const FEM& fem, const mfem::GridFunction& rho, const Args& args) {
  auto phi = std::make_unique<mfem::GridFunction>(fem.H1_fes.get());

  mfem::Array<int> ess_bdr(fem.mesh_ptr->bdr_attributes.Max());
  ess_bdr = 1;

  mfem::LinearForm mass_lf(fem.H1_fes.get());
  mfem::GridFunctionCoefficient rho_coeff(&rho);
  mass_lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(rho_coeff));
  mass_lf.Assemble();

  const double target_mass = mass_lf.Sum();
  // const double polar_potential = -G * target_mass / RADIUS;

  auto grav_potential = [&target_mass](const mfem::Vector& x) {
    const double r = x.Norml2();
    return (r > 1e-12) ? (-G * target_mass / r) : 0.0;
  };

  // mfem::ConstantCoefficient phi_bdr_coeff(polar_potential);
  mfem::FunctionCoefficient phi_bdr_coeff(grav_potential);
  mfem::Array<int> ess_tdof_list;
  fem.H1_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  MassContinuitySolver gravPotentialSolver(*fem.H1_fes);
  phi->ProjectBdrCoefficient(phi_bdr_coeff, ess_tdof_list);
  gravPotentialSolver.Solve(rho_coeff, *phi, ess_tdof_list);

  if (args.rotation) {
    auto rot = [&args](const mfem::Vector& x) {
      return centrifugal_potential(x, args.omega);
    };

    mfem::FunctionCoefficient centrifugal_coeff(rot);
    mfem::GridFunction centrifugal_gf(fem.H1_fes.get());
    centrifugal_gf.ProjectCoefficient(centrifugal_coeff);
    (*phi) += centrifugal_gf;
  }
  return phi;
}

void project_scalar_function(mfem::GridFunction& gf, const std::function<double(const mfem::Vector& x)> &g) {
  mfem::FunctionCoefficient coeff(g);
  gf.ProjectCoefficient(coeff);
}

std::unique_ptr<mfem::GridFunction> get_enthalpy(const FEM& fem, const mfem::GridFunction& phi) {
  const double polar_potential = get_polar_value(phi, *fem.mesh_ptr, RADIUS);

  auto H = std::make_unique<mfem::GridFunction>(fem.H1_fes.get());
  mfem::ConstantCoefficient bernoulli(polar_potential);

  mfem::GridFunctionCoefficient potentialCoeff(&phi);
  mfem::SumCoefficient enthalpyCoeff(bernoulli, potentialCoeff, 1.0, -1.0);
  
  H->ProjectCoefficient(enthalpyCoeff);

  return H;
}

double get_polar_value(const mfem::GridFunction& gf, mfem::Mesh& mesh, const double radius) {
  mfem::DenseMatrix pole(3, 1);
  pole(0, 0) = 0;
  pole(1, 0) = 0;
  pole(2, 0) = radius;

  mfem::Array<int> elementIds(1);
  mfem::Array<mfem::IntegrationPoint> ips(1);
  mesh.FindPoints(pole, elementIds, ips);

  if (elementIds[0] > 0) {
    const double value = gf.GetValue(elementIds[0], ips[0]);
    return value;
  } else {
    std::cerr << "Unable to locate pole..." << std::endl;
    exit(1);
  }
}

double gamma(double n) {
  if (n <= 0.0) {
    const std::string errMsg = std::format("polytropic index but be finite, non-zero, and positive. Got {}", n);
    std::cerr << errMsg << std::endl;
    exit(1);
  }

  return 1.0 + (1.0/n);

}

double rho_from_enthalpy_barotropic(double h, double n) {
  if (h <= 0.0) return 0.0;

  const double g = gamma(n);
  constexpr double K = 1.0;
  const double K_prime = (g * K) / (g - 1.0);

  return std::pow(h / K_prime, 1.0 / (g - 1.0));
}

double mix_density(const double rho_0, const double rho_1, const double alpha) {
  return alpha * rho_1 + (1-alpha) * rho_0;
}

std::unique_ptr<mfem::GridFunction> update_density(const FEM& fem, const FixedPoint& fp_old, const FixedPoint& fp_new, double n, double alpha) {
  mfem::GridFunctionCoefficient h_coeff(fp_new.h.get());

  auto trans = [&n](double h) {
    return rho_from_enthalpy_barotropic(h, n);
  };

  mfem::TransformedCoefficient rho_map_coeff(&h_coeff, trans);
  mfem::GridFunctionCoefficient rho_0_coeff(fp_old.rho.get());

  auto mixer = [&alpha](const double rho_0, const double rho_1) {
    return mix_density(rho_0, rho_1, alpha);
  };

  mfem::TransformedCoefficient rho_mixed(&rho_0_coeff, &rho_map_coeff, mixer);

  auto rho_new = std::make_unique<mfem::GridFunction>(fem.H1_fes.get());
  rho_new->ProjectCoefficient(rho_mixed);
  return rho_new;
}

std::unique_ptr<mfem::GridFunction> conserve_mass(const FEM& fem, const mfem::GridFunction& gf, double target_mass) {
  const double current_mass = get_current_mass(fem, gf);
  const double global_scaling_factor = target_mass / current_mass;

  auto new_rho = std::make_unique<mfem::GridFunction>(fem.H1_fes.get());
  mfem::GridFunctionCoefficient orig_rho_coeff(&gf);
  mfem::ConstantCoefficient scale_coeff(global_scaling_factor);

  mfem::ProductCoefficient scaled_rho(orig_rho_coeff, scale_coeff);


  new_rho->ProjectCoefficient(scaled_rho);

  return new_rho;
}

FixedPoint init_fp(const FEM& fem, const Args& args) {
  FixedPoint fp;
  fp.phi = get_potential(fem, *fem.rho_gf, args);
  fp.h = get_enthalpy(fem, *fp.phi);
  fp.rho = std::make_unique<mfem::GridFunction>(*fem.rho_gf);
  return fp;

}

FixedPoint step(const FEM& fem, const FixedPoint& fp, const Args& args) {
  FixedPoint nfp;
  nfp.phi = get_potential(fem, *fp.rho, args);
  nfp.h = get_enthalpy(fem, *nfp.phi);

  const auto new_density = update_density(fem, fp, nfp, args.index, args.alpha);
  nfp.rho = conserve_mass(fem, *new_density, MASS);

  return nfp;
}

void VisualizeFP(const FEM& fem, const FixedPoint& fp, const std::string& prefix) {
  auto titler = [&prefix](const std::string& name) -> std::string {
    return std::format("{}: {}", prefix, name);
  };
  ViewMesh(HOST, PORT, *fem.mesh_ptr, *fp.rho, titler("Density"));
  ViewMesh(HOST, PORT, *fem.mesh_ptr, *fp.phi, titler("Potential"));
  ViewMesh(HOST, PORT, *fem.mesh_ptr, *fp.h, titler("Enthalpy"));
}

double L2RelativeResidual(const FixedPoint& fp_old, const FixedPoint& fp_new) {
  mfem::GridFunctionCoefficient old_rho_coeff(fp_old.rho.get());

  const double l2_diff = fp_new.rho->ComputeL2Error(old_rho_coeff);
  const double l2_norm = fp_new.rho->Norml2();

  return (l2_norm > 1e-18) ? (l2_diff/l2_norm) : l2_diff;
}

std::expected<FixedPoint, FixedPointErrors> iterate_for_constant_shape(const FEM &fem, const Args &args) {
  FixedPoint fp = init_fp(fem, args);
  if (args.visualize && args.verbosity == Verbosity::VERBOSE) {VisualizeFP(fem, fp, "Initial");}

  for (int i = 0; i < args.max_iters; ++i) {
    FixedPoint new_fp = step(fem, fp, args);
    double err = L2RelativeResidual(fp, new_fp);

    if (args.verbosity >= Verbosity::PER_ITERATION) std::println("Iteration {:4} -- ||r|| = {:7.5E}", i, err);

    fp = new_fp.clone();
    if (args.visualize && args.verbosity == Verbosity::VERBOSE) {VisualizeFP(fem, fp, std::format("Step Number {}", i));}
    if (err <= args.tol) {
      if (args.verbosity >= Verbosity::PER_ITERATION) std::println("Convergence reached in {} steps!", i);
      break;
    }
  }

  if (args.visualize) {VisualizeFP(fem, fp, "Final");}
  return fp;
}

double clip(const double h, const double relax) {
  return std::max(h, 0.0) * relax;
}

void radial(const mfem::Vector& x, mfem::Vector &r_hat) {
  r_hat = x;
  if (const double norm = r_hat.Norml2(); norm > 1e-12) {
    r_hat /= norm;
  } else {
    r_hat = 0.0;
  }
}

bool is_system_bound(const FEM& fem, const FixedPoint& fp, const Args& args) {
    mfem::GridFunctionCoefficient rho_coeff(fp.rho.get());
    mfem::GridFunctionCoefficient h_coeff(fp.h.get());

    auto cent_func = [&args](const mfem::Vector& x) {
        return centrifugal_potential(x, args.omega);
    };
    mfem::FunctionCoefficient cent_coeff(cent_func);

    mfem::GridFunction phi_grav_gf(*fp.phi);
    mfem::GridFunction cent_gf(fem.H1_fes.get());
    cent_gf.ProjectCoefficient(cent_coeff);
    phi_grav_gf -= cent_gf;

    mfem::GridFunctionCoefficient phi_grav_coeff(&phi_grav_gf);

    mfem::ProductCoefficient rho_phi_coeff(rho_coeff, phi_grav_coeff);
    mfem::LinearForm w_lf(fem.H1_fes.get());
    w_lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(rho_phi_coeff));
    w_lf.Assemble();
    double W = 0.5 * w_lf.Sum();

    mfem::ProductCoefficient rho_cent_coeff(rho_coeff, cent_coeff);
    mfem::LinearForm t_lf(fem.H1_fes.get());
    t_lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(rho_cent_coeff));
    t_lf.Assemble();
    double T = -1.0 * t_lf.Sum();

    mfem::ProductCoefficient rho_h_coeff(rho_coeff, h_coeff);
    mfem::LinearForm u_lf(fem.H1_fes.get());
    u_lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(rho_h_coeff));
    u_lf.Assemble();
    double integral_rho_h = u_lf.Sum();
    double U = (args.index / (args.index + 1.0)) * integral_rho_h;

    double E_total = W + T + U;

    if (args.verbosity >= Verbosity::FULL) {
        std::println("--- Energy Diagnostics ---");
        std::println("  W (Grav) : {:+.4E}", W);
        std::println("  T (Rot)  : {:+.4E}", T);
        std::println("  U (Int)  : {:+.4E}", U);
        std::println("  Total E  : {:+.4E}", E_total);
        std::println("--------------------------");
    }

    return E_total < 0.0;
}

std::unique_ptr<mfem::GridFunction> get_nodal_displacement(const FEM& fem, const FixedPoint &fp, const Args& args) {
  auto displacement = std::make_unique<mfem::GridFunction>(fem.Vec_H1_fes.get());
  *displacement = 0.0;

  mfem::GridFunctionCoefficient h_coeff(fp.h.get());
  mfem::TransformedCoefficient mag_coeff(&h_coeff, [&args](double h){ return h * args.relax_rate; });
  mfem::VectorFunctionCoefficient dir_coeff(fem.mesh_ptr->Dimension(), radial);
  mfem::ScalarVectorProductCoefficient total_dist_coeff(mag_coeff, dir_coeff);

  displacement->ProjectCoefficient(total_dist_coeff);

  mfem::BilinearForm a(fem.Vec_H1_fes.get());
  a.AddDomainIntegrator(new mfem::VectorDiffusionIntegrator());
  a.Assemble();

  mfem::Array<int> ess_bdr(fem.mesh_ptr->bdr_attributes.Max());
  ess_bdr = 1; // Mark the outer surface
  mfem::Array<int> ess_tdof_list;
  fem.Vec_H1_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  mfem::LinearForm b(fem.Vec_H1_fes.get());
  b.Assemble();

  mfem::OperatorPtr A;
  mfem::Vector B, X;
  a.FormLinearSystem(ess_tdof_list, *displacement, b, A, X, B);

  mfem::CGSolver cg;
  cg.SetRelTol(1e-6);
  cg.SetMaxIter(500);
  cg.SetOperator(*A);
  cg.Mult(B, X);

  a.RecoverFEMSolution(X, b, *displacement);

  return displacement;
}

void deform_mesh(const FEM& fem, const mfem::GridFunction& displacement, const Args& args) {
  if (args.allow_deformation) {
    fem.mesh_ptr->MoveNodes(displacement);
    fem.mesh_ptr->NodesUpdated();
  }
}

void print_options(const Args& args) {
  std::println("Options:");
  std::println("  Visualize: {}", args.visualize);
  std::println("  Rotation: {}", args.rotation);
  std::println("  Polytropic Index: {}", args.index);
  std::println("  Density Mixing Alpha: {}", args.alpha);
  std::println("  Mesh File: {}", args.mesh);
  std::println("  Max Iterations: {}", args.max_iters);
  std::println("  Tolerance: {:5.2E}", args.tol);
  std::println("  Angular Velocity (Omega): {}", args.omega);
  std::println("  Relaxation Rate: {}", args.relax_rate);
  std::println("  Allow Deformation: {}", args.allow_deformation);
  std::println("  Max Deformation Iterations: {}", args.max_deformation_iters);
  std::println("  Deformation Relative Tolerance: {:5.2E}", args.deformation_rtol);
  std::println("  Deformation Absolute Tolerance: {:5.2E}", args.deformation_atol);
}

Args setup_cli(const int argc, char** argv) {
  Args args;

  CLI::App app{"Simple Potential, Enthalpy, and Pressure Solver"};

  bool no_rotate = false;
  bool no_deform = false;

  app.add_flag("-v,--visualize", args.visualize, "Enable Visualizations");
  app.add_option("-n,--index", args.index, "polytropic index");
  app.add_option("-a,--alpha", args.alpha, "density mixing strength");
  app.add_option("-m,--mesh", args.mesh, "mesh file to read from");
  app.add_option("-i,--max-iters", args.max_iters, "Max number of iterations");
  app.add_option("-t,--tol", args.tol, "Relative tolerance to end on");
  app.add_flag("--no-rotate", no_rotate, "Enable or disable rotation");
  app.add_option("-w,--omega", args.omega, "Arbitrary Unit Angular Velocity");
  app.add_option("-r,--relax-rate", args.relax_rate, "Relaxation rate for boundary displacement");
  app.add_flag("--no-deform", no_deform, "Enable or disable mesh deformation");
  app.add_option("--max-deformation-iters", args.max_deformation_iters, "Max number of deformation steps to take");
  app.add_option("--deformation-rtol", args.deformation_rtol, "Relative tolerance for deformation convergence");
  app.add_option("--deformation-atol", args.deformation_atol, "Absolute tolerance for deformation convergence");

  const auto verbosity_map = make_verbosity_map();

  app.add_option("--verbosity", args.verbosity, "Set the verbosity level")->transform(CLI::CheckedTransformer(verbosity_map, CLI::ignore_case));

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    exit(app.exit(e));
  }

  args.rotation = !no_rotate;
  args.allow_deformation = !no_deform;

  return args;
}

Envelope extract_envelope(const FEM& fem, const Args& args) {
  Envelope env;
  for (size_t i = 0; i < fem.mesh_ptr->GetNBE(); ++i) {
    mfem::ElementTransformation *trans = fem.mesh_ptr->GetBdrElementTransformation(static_cast<int>(i));
    const mfem::Geometry::Type geom = fem.mesh_ptr->GetBdrElementGeometry(static_cast<int>(i));
    const mfem::RefinedGeometry *refiner = mfem::GeometryRefiner().Refine(geom, args.env_ref_levels);
    const mfem::IntegrationRule *ir = &refiner->RefPts;

    for (size_t j = 0; j < ir->GetNPoints(); ++j) {
      const mfem::IntegrationPoint &ip = ir->IntPoint(static_cast<int>(j));

      mfem::Vector phys_point;
      trans->Transform(ip, phys_point);

      Point p{
        .x = phys_point(0),
        .y = phys_point(1),
        .z = phys_point(2),
        .r = phys_point.Norml2(),
        .IR_ID = j,
        .BE_ID = i

      };

      env.points.push_back(p);
    }
  }
  return env;
}

std::map<std::string, Verbosity> make_verbosity_map() {
  std::map<std::string, Verbosity> verbosity_map;
  for (const auto& [key, pair] : VerbosityNames) {
    verbosity_map[pair.first] = key;
    verbosity_map[pair.second] = key;
  }
  return verbosity_map;
}
