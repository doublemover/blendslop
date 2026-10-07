"""A process-owned numeric helper with bounded file IPC and explicit shutdown."""
from __future__ import annotations
import atexit,json,os,pickle,subprocess,tempfile,time,uuid
from pathlib import Path


def _atomic_packet(path,value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(value,stream,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary,path)


class NumericHelperSession:
    def __init__(self,executable,script,*,source_paths=(),idle_timeout_s=60.):
        # A venv interpreter commonly symlinks the system binary. Resolving it
        # for invocation discards pyvenv.cfg and imports the wrong environment.
        self.executable=Path(os.path.abspath(executable))
        if not self.executable.is_file():
            raise FileNotFoundError(self.executable)
        self.script=Path(script).resolve(strict=True)
        paths=[Path(p).resolve(strict=True) for p in source_paths]
        configuration=self.executable.parent.parent/'pyvenv.cfg'
        if configuration.is_file():
            paths.append(configuration)
        self.source_paths=tuple(paths)
        self.idle_timeout_s=float(idle_timeout_s)
        self.directory=tempfile.TemporaryDirectory(prefix='blendslop-owned-numeric-')
        self.root=Path(self.directory.name);self.process=None;self.log=None
        self.identity=None;self.starts=0;self.closed=False

    def _signature(self):
        result=[]
        for path in (self.executable,self.script,*self.source_paths):
            stat=path.stat()
            result.append((str(path),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns))
        return tuple(result)

    def _stop(self):
        if self.process is not None and self.process.poll() is None:
            (self.root/'stop').touch()
            try:
                self.process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=1.)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.process=None
        if self.log is not None:
            self.log.close();self.log=None

    def _ensure(self,deadline):
        if self.closed:
            raise RuntimeError('numeric helper owner is closed')
        signature=self._signature()
        restarted=self.process is None or self.process.poll() is not None or self.identity!=signature
        if restarted:
            self._stop()
            for path in (self.root/'ready.json',self.root/'stop'):
                path.unlink(missing_ok=True)
            (self.root/'heartbeat').touch()
            self.log=(self.root/'worker.log').open('ab')
            # Host image dependencies can contain extension wheels for another
            # interpreter. Numeric children use their selected environment only.
            environment=dict(os.environ)
            environment.pop('PYTHONPATH',None);environment.pop('PYTHONHOME',None)
            self.process=subprocess.Popen([str(self.executable),'-B',str(self.script),'--serve',
                '--directory',str(self.root),'--idle-timeout',str(self.idle_timeout_s)],
                stdin=subprocess.DEVNULL,stdout=self.log,stderr=subprocess.STDOUT,env=environment,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.identity=signature;self.starts+=1
        while not (self.root/'ready.json').is_file():
            if time.monotonic()>=deadline:
                self._stop();raise TimeoutError('numeric helper startup allowance exhausted')
            if self.process.poll() is not None:
                self._stop();raise RuntimeError('numeric helper exited before ready')
            (self.root/'heartbeat').touch();time.sleep(.01)
        return json.loads((self.root/'ready.json').read_text()),restarted

    def call(self,payload=None,*,timeout_s=10.):
        started=time.monotonic();deadline=started+max(.001,float(timeout_s))
        ready,restarted=self._ensure(deadline)
        if payload is None:
            return {**ready,'helper_session':{'pid':self.process.pid,'starts':self.starts,
                'restarted':restarted,'warm_owned_session':True}}
        job=uuid.uuid4().hex
        output=self.root/(job+'.output.pkl');progress=self.root/(job+'.progress.pkl')
        forwarded=dict(payload)
        forwarded['progress_paths']=[*forwarded.get('progress_paths',()),str(progress)]
        forwarded['timeout_s']=min(float(forwarded.get('timeout_s') or timeout_s),max(.001,deadline-time.monotonic()))
        input_path=self.root/(job+'.input.pkl')
        _atomic_packet(input_path,forwarded)
        while not output.is_file():
            (self.root/'heartbeat').touch()
            if time.monotonic()>=deadline or self.process.poll() is not None:
                self._stop()
                input_path.unlink(missing_ok=True)
                output.unlink(missing_ok=True)
                if progress.is_file():
                    with progress.open('rb') as stream:retained=pickle.load(stream)['value']
                    progress.unlink(missing_ok=True)
                    return {**retained,'partial':True,'stop_reason':'owned_helper_interrupted',
                            'final_update_evaluated':False,'helper_session':{'starts':self.starts}}
                raise TimeoutError('numeric helper interrupted without a scored checkpoint')
            time.sleep(.01)
        with output.open('rb') as stream:response=pickle.load(stream)
        output.unlink();progress.unlink(missing_ok=True)
        if response.get('job')!=job:
            raise ValueError('numeric helper response belongs to a different job')
        if response.get('status')!='success':
            raise RuntimeError('numeric helper job failed: '+str(response.get('error')))
        result=response['value']
        result['helper_session']={'pid':self.process.pid,'starts':self.starts,'warm_owned_session':True,
            'restarted':restarted,'request_wall_s':time.monotonic()-started}
        return result

    def close(self):
        if not self.closed:
            self._stop();self.directory.cleanup();self.closed=True

    def __enter__(self):return self
    def __exit__(self,*_):self.close()


_SESSIONS={}


def dvx_session(executable):
    root=Path(__file__).resolve().parents[3]
    script=root/'scripts/dvx_worker.py'
    sources=[Path(__file__).with_name('dvx_adapter.py'),Path(__file__).with_name('conditioned_solver.py'),
             Path(__file__).with_name('mesh_conditioning.py'),Path(__file__).with_name('projected_mesh_rays.py')]
    key=os.path.abspath(executable)
    if key not in _SESSIONS or _SESSIONS[key].closed:
        _SESSIONS[key]=NumericHelperSession(key,script,source_paths=sources)
    return _SESSIONS[key]


def close_helper_sessions():
    for session in list(_SESSIONS.values()):session.close()
    _SESSIONS.clear()


atexit.register(close_helper_sessions)
